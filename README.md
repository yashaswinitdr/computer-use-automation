# Computer-Use Automation System

An LLM discovers how to do a task in a legacy back-office UI once. The run is recorded as a typed, versioned
**capability** artifact. After that, the capability is replayed **deterministically, with no model in the loop**. Runtime
errors are handled explicitly, risky steps are gated by a human, and a human operator can take over the live session.

```
goal ──► LLM discovery (observe → decide → act) ──► capability artifact (JSON, reviewed, approved)
                                                          │
            agent / caller ──► replay(capability, inputs) ─┴─► succeeded(outputs) | business_outcome(code) | failed(code, evidence)
                                     ▲        │
                         human operator ◄─────┘ intervention request (stuck / approval) → take control → resume
```

The design write-up is in [REPORT.md](REPORT.md). Evidence from real runs is in [evidence/](evidence/README.md).

## Layout

| path | what |
|---|---|
| `cua/schema.py` | Capability / AppProfile / TenantConfig / ReplayResult models (the contract) |
| `cua/surface/` | perception + action seam (`base.py` protocol, `web.py` Playwright implementation for legacy markup) |
| `cua/agent/` | discovery loop (`discovery.py`), provider-neutral LLM adapters (`llm.py`) |
| `cua/recorder.py` | discovery trace → parameterised capability |
| `cua/replay.py` | deterministic replay engine and runtime-state handling |
| `cua/handoff.py`, `cua/operator_console.py` | escalation, control transfer, operator console |
| `cua/policy.py`, `cua/redact.py`, `policy.json` | allowlist, risk classes, redaction |
| `profiles/` | app profile per vendor product version (sign-on flow + runtime-state taxonomy) |
| `tenants/` | per-institution config (base URL, secret refs, caption overlays) |
| `capabilities/` | recorded artifacts (`<id>@<version>.json`) |
| `target_app/` | the proxy target: a deliberately hostile, synthetic legacy core-banking app, with fault injection |
| `evidence/` | discovery + replay runs (logs, masked screenshots, results) |

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

```bash
python -m playwright install chromium
```

```bash
cp .env.example .env
```

Discovery needs an LLM. Pick one:

- **Local, no API key (used for the evidence in this repo):** install [Ollama](https://ollama.com), then run
  `ollama pull qwen2.5:7b` and pass `--provider ollama --model qwen2.5:7b`. Screen data never leaves the machine.
  Set `OLLAMA_HOST` if the server isn't on the default `http://127.0.0.1:11434`.
- **Hosted:** put `ANTHROPIC_API_KEY` (default provider, model `claude-opus-5`) or `OPENAI_API_KEY` (model
  `gpt-4.1`) in `.env`.

Override the choice with `--provider`/`--model` or `CUA_PROVIDER`/`CUA_MODEL`. Only discovery uses a model. Replay
and all tests run without any live service. The teller credentials in `.env.example` are test values for the local
synthetic app (see `target_app/seed.py`).

Start the target app. Tenant A is on :5055. Tenant B runs "the same product, configured differently" on :5057.

```bash
python -m cua app --variant a --port 5055
```

```bash
python -m cua app --variant b --port 5057
```

## Demo path

**1. Discover** (real LLM run). This records `capabilities/coreteller.member.savings_balance@1.0.0.json`.

```bash
python -m cua discover --provider ollama --model qwen2.5:7b --goal "Look up member {member_id} and read their current share savings balance" --param member_id=10042 --id coreteller.member.savings_balance --headed
```

**2. Review and approve.** New artifacts are `draft`, and unattended replay requires `approved`.

```bash
python -m cua show coreteller.member.savings_balance
```

```bash
python -m cua approve capabilities/coreteller.member.savings_balance@1.0.0.json --reviewer you
```

**3. Replay** deterministically with new inputs (no LLM):

```bash
python -m cua replay coreteller.member.savings_balance --param member_id=20077
```

**4. Error and outcome handling.** These cases cover a business outcome, recoverable faults, a hard failure, and a
cross-tenant replay:

```bash
python -m cua replay coreteller.member.savings_balance --param member_id=99999
```

```bash
python -m cua replay coreteller.member.savings_balance --param member_id=10042 --inject session_expire=1 after=2
```

```bash
python -m cua replay coreteller.member.savings_balance --param member_id=10042 --inject abend=1 after=2 --no-escalate
```

```bash
python -m cua replay coreteller.member.savings_balance --param member_id=10042 --tenant tenants/rvfcu.json
```

**5. Human takeover of the live session.** Inject a screen the system has never seen. Replay escalates and prints
the operator console URL (http://127.0.0.1:5056). In the console, click **Take control**. Work in the visible
browser window (for example, click "Remind Me Later"), then click **Resume**. The run re-verifies state and finishes.

```bash
python -m cua replay coreteller.member.savings_balance --param member_id=10042 --inject survey=1 after=1 --headed
```

The operator side also works from a second terminal (`python -m cua operator list|claim|resolve ...`). Demos and
tests use `--operator-script` (a scripted stand-in that drives the same live page with real input events).

**6. Agent-facing catalog** (stretch goal). Approved capabilities are exposed as tool definitions and can be invoked
by name:

```bash
python -m cua catalog
```

```bash
python -m cua invoke coreteller__member__savings_balance --args "{\"member_id\": \"20077\"}"
```

**Everything at once.** This regenerates `evidence/`: both discoveries, then 15 replays covering every
outcome class.

```bash
python scripts/demo.py
```

## Tests

The suite has 21 tests. It includes 12 end-to-end tests that drive a real Chromium against the app in-process: discovery (with a
scripted model stand-in), recording, replay, every result class, recoveries, approval gating, human handoff, and cross-tenant
overlays. No API key is needed.

```bash
python -m pytest -q
```

## What is real and what is mocked

- **Real:** the LLM discovery runs, browser automation, artifact recording, replay, policy enforcement (including
  the network-level allowlist), redaction, the handoff state machine and live-session takeover, human-action
  capture, and the operator console.
- **The two discovery runs in `evidence/`** were driven by Qwen 2.5 7B running locally under Ollama:
  - **Savings balance (`01`):** completed autonomously. Every step was a model decision, and the capability that
    every later replay uses was recorded from this run.
  - **Open sub-account (`14`):** this longer multi-field flow was beyond the small local model. It got stuck, a
    human operator took control once and handed back, and it got stuck again. The run is kept as evidence of stuck
    detection and handoff during discovery. So that the write-flow replay paths (approval, denial, host validation)
    can still be shown, that capability was recorded with the scripted stand-in (`15`) and is labelled as such,
    including `provenance.model = "scripted"`. A stronger hosted model should be used to re-record it.
- **Mocked, deliberately:**
  - The target is a local synthetic app, because we must not touch a real bank system.
  - `SimulatedOperator` stands in for a person in demos and tests. It acts through real input events on the live
    page, so capture is exercised for real.
  - `ScriptedLLM` replaces the model in tests, and in recording `15` as described above.
- **Designed but not built:** desktop/UIA and pixel-only surfaces, streamed co-browsing console, multi-tenant
  registry service. See REPORT.md §4 and §7.
