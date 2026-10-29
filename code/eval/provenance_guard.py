"""Argument-level provenance guard.

The guard sits at the tool-call boundary. For every action with a
value-bearing argument (a payment recipient, an invitee address, a firewall
source IP, a route next-hop, an expense category), it checks where each
value came from: a value is trusted if it already appeared, earlier in the
run, in the user's own request or in the output of a tool the deployment
marks as trusted (the directory, the expense policy, IPAM, the device
inventory). A call whose value appears only in untrusted content is held
for confirmation instead of executed.

The guard needs only two pieces of configuration -- which tools are trusted
and which arguments carry values -- and no task-specific knowledge, attack
signature or prompt change. It is deterministic and adds no model calls.
"""

import re


# Tools whose output the deploying organization controls.
TRUSTED_TOOLS = {"lookup_contact_email", "lookup_expense_policy", "lookup_ipam", "get_device_inventory"}


# Value-bearing arguments of each action tool.
GUARDED_ARGS = {
    "send_money": ["recipient"],
    "invite_user_to_slack": ["user_email"],
    "book_calendar_event": ["attendees"],
    "log_expense": ["category"],
    "apply_acl_rule": ["src_ip"],
    "set_static_route": ["next_hop"],
}


def _text(content) -> str:
    if isinstance(content, list):
        return " ".join(str(c.get("content", "")) for c in content if isinstance(c, dict))
    return "" if content is None else str(content)


def _value_pattern(value: str) -> str:
    """Whole-value pattern (lowercase): 10.0.2.5 does not match inside
    10.0.2.51, nor john inside john.smith@..., but a trailing sentence
    period is allowed."""
    return r"(?<![\w.@-])" + re.escape(value.strip().lower()) + r"(?![\w@-]|\.\w)"


def value_in_text(value: str, text: str) -> bool:
    if not value.strip():
        return False  # an empty pattern matches any text; a blank value is never "seen"
    return re.search(_value_pattern(value), text.lower()) is not None


def _values(arg) -> list[str]:
    if isinstance(arg, (list, tuple, set)):
        return [str(v) for v in arg]
    return [] if arg is None else [str(arg)]


def check_trace(messages: list[dict], verifier=None) -> list[dict]:
    """Guard decision for every guarded call in a trace, in order. Trusted
    text grows as the run proceeds, so only outputs that precede a call can
    vouch for it. With a `verifier` (eval/signed_trust.py), a trusted-tool
    output vouches only if its signature verifies; without one, the tool
    name alone decides."""
    trusted_text = ""
    decisions = []
    for m in messages:
        if m["role"] == "user":
            trusted_text += " " + _text(m.get("content"))
        elif m["role"] == "assistant":
            for call in m.get("tool_calls") or []:
                args = call.get("args") or {}
                for arg_name in GUARDED_ARGS.get(call["function"], []):
                    values = _values(args.get(arg_name))
                    untrusted = [v for v in values if not value_in_text(v, trusted_text)]
                    decisions.append({
                        "function": call["function"],
                        "arg": arg_name,
                        "values": values,
                        "untrusted_values": untrusted,
                        "decision": "HOLD" if untrusted or not values else "ALLOW",
                    })
        elif m["role"] == "tool":
            call = m.get("tool_call") or {}
            if call.get("function") in TRUSTED_TOOLS and (verifier is None or verifier(m)):
                # A trusted tool vouches for what it returns, not for the
                # query it was given: "No entry found for '<x>'" must not
                # make <x> trusted.
                output = _text(m.get("content"))
                for arg in (call.get("args") or {}).values():
                    for v in _values(arg):
                        if v.strip():
                            output = re.sub(_value_pattern(v), " ", output.lower())
                trusted_text += " " + output
    return decisions
