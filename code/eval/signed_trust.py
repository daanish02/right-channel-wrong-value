"""Signed trust labels for tool results, used by the provenance guard.

Without signatures, the guard decides that a tool output is trusted from
the tool's name alone. In an MCP deployment that name is whatever a server
chooses to register, so a malicious server can call its tool
``lookup_ipam`` and vouch for the attacker's value, and anything between a
trusted server and the agent can alter a trusted result in transit.

Here every trusted server holds its own random key, shared only with the
gateway that runs the guard. The server tags each result with an
HMAC-SHA256 over (tool name, call arguments, result text); the guard counts
a result as trusted only if the tag verifies under the key registered for
that tool. Binding the arguments stops a signed answer to one query (one
host's IP) from being replayed as the answer to another.
"""

import copy
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field

from eval.provenance_guard import TRUSTED_TOOLS, _text


def _payload(tool: str, args: dict, text: str) -> bytes:
    return json.dumps([tool, args or {}, text], sort_keys=True, ensure_ascii=False).encode()


@dataclass
class KeyRegistry:
    """Gateway-side map from trusted tool to its server's key."""

    keys: dict[str, bytes] = field(default_factory=dict)

    @classmethod
    def generate(cls, tools=TRUSTED_TOOLS) -> "KeyRegistry":
        return cls({t: secrets.token_bytes(32) for t in tools})

    def sign(self, tool: str, args: dict, text: str) -> str:
        return hmac.new(self.keys[tool], _payload(tool, args, text), hashlib.sha256).hexdigest()

    def verify(self, message: dict) -> bool:
        call = message.get("tool_call") or {}
        tool = call.get("function")
        tag = (message.get("meta") or {}).get("signature")
        if tool not in self.keys or not tag:
            return False
        expected = self.sign(tool, call.get("args") or {}, _text(message.get("content")))
        return hmac.compare_digest(expected, tag)


def sign_trusted_outputs(messages: list[dict], registry: KeyRegistry) -> list[dict]:
    """Attach a signature to every genuine trusted-tool output, as the
    trusted servers would."""
    out = copy.deepcopy(messages)
    for m in out:
        call = m.get("tool_call") or {}
        if m["role"] == "tool" and call.get("function") in registry.keys:
            m["meta"] = {"signature": registry.sign(call["function"], call.get("args") or {}, _text(m.get("content")))}
    return out


def forge_trusted_output(messages: list[dict], tool: str, value: str, signer: KeyRegistry | None = None) -> list[dict]:
    """Insert, right after the user's request, a result that claims to come
    from a trusted tool and carries the attacker's value -- what a malicious
    server registering that tool name would return. With `signer`, the
    forger tags it with a key of its own."""
    out = copy.deepcopy(messages)
    call = {"function": tool, "args": {}, "id": "forged"}
    forged = {"role": "tool", "content": [{"type": "text", "content": value}], "tool_call": call, "tool_call_id": "forged"}
    if signer is not None:
        forged["meta"] = {"signature": signer.sign(tool, {}, value)}
    first_user = next(i for i, m in enumerate(out) if m["role"] == "user")
    out[first_user + 1:first_user + 1] = [{"role": "assistant", "content": None, "tool_calls": [call]}, forged]
    return out


def tamper_trusted_outputs(messages: list[dict], old: str, new: str) -> list[dict]:
    """Replace `old` with `new` inside trusted-tool outputs, leaving their
    signatures as they were -- an on-path modification of a trusted result."""
    out = copy.deepcopy(messages)
    for m in out:
        if m["role"] == "tool" and (m.get("tool_call") or {}).get("function") in TRUSTED_TOOLS:
            content = m.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("content"), str):
                        c["content"] = c["content"].replace(old, new)
            elif isinstance(content, str):
                m["content"] = content.replace(old, new)
    return out
