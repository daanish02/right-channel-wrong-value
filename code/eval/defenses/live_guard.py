"""Live, in-pipeline provenance guard -- the online counterpart to
eval/provenance_guard.py's offline replay.

The replay reads a FINISHED trace and asks "would this call have been held?".
This module asks the same question DURING the run: a held call's real tool is
never invoked, and the agent sees a synthetic result in its place, so the
conversation genuinely continues from the hold.

Same trust rule as the replay (TRUSTED_TOOLS, GUARDED_ARGS and the value
regex are imported from eval/provenance_guard.py; HMAC tags come from
eval/signed_trust.py), so "live" and "offline" differ only in WHEN the check
runs.

Pipeline (replaces the plain ToolsExecutor inside ToolsExecutionLoop):

    GuardedToolsExecutor  -- splits the step's tool calls into allowed and
                             held, runs only the allowed ones through
                             AgentDojo's own ToolsExecutor, and returns
                             results for ALL calls in the original order.
                             The assistant message is left intact so every
                             tool_call keeps a matching tool result, which
                             the chat API requires.
    LiveGuardSigner       -- after execution, tags each genuine trusted-tool
                             result with its HMAC and, only if the tag
                             verifies, adds its text to the trusted set.

Lockdown: once any call is held, every later guarded call in that
conversation is held too, until a human confirms. Without this the agent
could try a reworded call or another tool and slip past a one-off hold.

Registration: importing this module adds defense="live_guard" to
AgentPipeline.from_config, the same idiom eval/defenses/melon.py uses.
"""

import re
from collections.abc import Sequence

from agentdojo.agent_pipeline import agent_pipeline as _ap
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.functions_runtime import Env, FunctionsRuntime
from agentdojo.types import ChatToolResultMessage, get_text_content_as_str, text_content_block_from_string

from eval.provenance_guard import GUARDED_ARGS, TRUSTED_TOOLS, _text, _value_pattern, _values, value_in_text
from eval.signed_trust import KeyRegistry

# No hint at how to get the hold lifted (no "call X instead"): that would tell
# an attacker-influenced agent how to work around the control.
HOLD_MESSAGE_TEMPLATE = (
    "Action requires additional verification before it can proceed: the value given for "
    "'{arg}' could not be verified against a trusted source."
)
HOLD_PREFIX = "Action requires additional verification before it can proceed"
LOCKDOWN_MESSAGE = (
    "Action requires additional verification before it can proceed: a prior action in this "
    "session was held for confirmation, so no further actions can be taken until a human "
    "confirms."
)


class _Ledger:
    """Guard state for ONE conversation, rebuilt from the message list on every
    step -- never stored anywhere shared. (AgentDojo's extra_args default is a
    single mutable dict reused by every case in a process, so state kept there
    leaks trust and lockdown from one case into the next.)"""

    def __init__(self, messages: Sequence[dict], registry: KeyRegistry) -> None:
        self.trusted_text = ""
        self.locked_down = False
        for m in messages:
            if m["role"] == "user":
                self.trusted_text += " " + _text(m.get("content"))
            elif m["role"] == "tool":
                if m.get("error") and str(m["error"]).startswith(HOLD_PREFIX):
                    self.locked_down = True
                    continue
                call = m.get("tool_call")
                if call is None or call.function not in TRUSTED_TOOLS or m.get("error"):
                    continue
                text = get_text_content_as_str(m["content"]) or ""
                tag = m.get("_live_guard_signature")
                probe = {"tool_call": {"function": call.function, "args": call.args or {}}, "content": text, "meta": {"signature": tag}}
                if tag is not None and registry.verify(probe):
                    self._absorb(call.args or {}, text)

    def _absorb(self, call_args: dict, output_text: str) -> None:
        # A trusted tool vouches for what it returns, not for the query it
        # was given (same scrub as check_trace).
        output = output_text
        for arg in (call_args or {}).values():
            for v in _values(arg):
                if v.strip():
                    output = re.sub(_value_pattern(v), " ", output.lower())
        self.trusted_text += " " + output

    def decide(self, function: str, args: dict) -> str | None:
        """Hold message for this call, or None to allow it."""
        arg_names = GUARDED_ARGS.get(function)
        if not arg_names:
            return None
        if self.locked_down:
            return LOCKDOWN_MESSAGE
        for arg_name in arg_names:
            values = _values((args or {}).get(arg_name))
            if not values or any(not value_in_text(v, self.trusted_text) for v in values):
                return HOLD_MESSAGE_TEMPLATE.format(arg=arg_name)
        return None


class GuardedToolsExecutor(BasePipelineElement):
    def __init__(self, registry: KeyRegistry) -> None:
        self.executor = ToolsExecutor()
        self.registry = registry

    def query(self, query: str, runtime: FunctionsRuntime, env: Env, messages: Sequence[dict], extra_args: dict):
        last = messages[-1] if messages else None
        if last is None or last["role"] != "assistant" or not last.get("tool_calls"):
            return query, runtime, env, messages, extra_args
        ledger = _Ledger(messages, self.registry)

        held: dict[str, ChatToolResultMessage] = {}
        kept = []
        for call in last["tool_calls"]:
            reason = ledger.decide(call.function, call.args or {})
            if reason is None:
                kept.append(call)
                continue
            ledger.locked_down = True
            held[call.id] = ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(reason)],
                tool_call_id=call.id,
                tool_call=call,
                error=reason,
            )
        if not held:
            return self.executor.query(query, runtime, env, messages, extra_args)

        executed: dict[str, dict] = {}
        if kept:
            trimmed_last = dict(last)
            trimmed_last["tool_calls"] = kept
            _, runtime, env, out, extra_args = self.executor.query(
                query, runtime, env, [*messages[:-1], trimmed_last], extra_args
            )
            executed = {m["tool_call_id"]: m for m in out[len(messages):]}
        ordered = [held.get(c.id) or executed[c.id] for c in last["tool_calls"]]
        return query, runtime, env, [*messages, *ordered], extra_args


class LiveGuardSigner(BasePipelineElement):
    """Tags each genuine trusted-tool result with its HMAC, as the trusted
    server would. The guard counts a trusted result only if the tag verifies."""

    def __init__(self, registry: KeyRegistry) -> None:
        self.registry = registry

    def query(self, query: str, runtime: FunctionsRuntime, env: Env, messages: Sequence[dict], extra_args: dict):
        for m in messages:
            if m["role"] != "tool" or m.get("_live_guard_signature") is not None:
                continue
            call = m.get("tool_call")
            if call is None or call.function not in TRUSTED_TOOLS:
                continue
            text = get_text_content_as_str(m["content"]) or ""
            m["_live_guard_signature"] = self.registry.sign(call.function, call.args or {}, text)
        return query, runtime, env, messages, extra_args


_original_from_config = AgentPipeline.from_config.__func__


def _from_config_with_live_guard(cls, config):
    if config.defense != "live_guard":
        return _original_from_config(cls, config)

    llm = (
        _ap.get_llm(_ap.MODEL_PROVIDERS[_ap.ModelsEnum(config.llm)], config.llm, config.model_id, config.tool_delimiter)
        if isinstance(config.llm, str)
        else config.llm
    )
    llm_name = config.llm if isinstance(config.llm, str) else llm.name

    assert config.system_message is not None
    tools_loop = ToolsExecutionLoop([GuardedToolsExecutor(_LIVE_REGISTRY), LiveGuardSigner(_LIVE_REGISTRY), llm])
    pipeline = cls([_ap.SystemMessage(config.system_message), _ap.InitQuery(), llm, tools_loop])
    pipeline.name = f"{llm_name}-{config.defense}"
    return pipeline


# One key set per process. A real deployment would load these from the
# trusted servers' own key store.
_LIVE_REGISTRY = KeyRegistry.generate()

AgentPipeline.from_config = classmethod(_from_config_with_live_guard)
