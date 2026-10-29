# Right Channel, Wrong Value

An argument-level provenance guard for tool-using LLM agents, with signed trust labels. The guard lets an action run
only if every value it carries appeared in the user's request or in a signed result from a trusted tool.

## Layout

```
code/eval/provenance_guard.py       the guard rule: was this value seen in the request or a trusted result?
code/eval/signed_trust.py           HMAC trust labels for trusted tool results
code/eval/defenses/live_guard.py    the guard in front of an AgentDojo agent
code/tests/                         unit tests
```

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). From `code/`:

```
uv sync
uv run pytest -q tests
```

## License

MIT, see `LICENSE`. Built on [AgentDojo](https://github.com/ethz-spylab/agentdojo).
