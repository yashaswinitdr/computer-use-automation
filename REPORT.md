# Design report

The proxy target is a synthetic "CoreTeller 4.2" teller app I wrote to be hostile in the ways real core-banking screens are. It uses a frameset (nav frame plus main frame), table layouts with captions in sibling `<td>`s, opaque field names (`f1`, `q`, `x9`), no ids, test ids or ARIA, and JavaScript-submitted buttons. Errors are red `<font>` text. It also has fault injection for interstitials, session expiry, host ABENDs, slow responses, and a screen the system has never seen. Two flows are implemented: member lookup (search → detail → read), and opening a sub-account (a multi-field form → review → irreversible **Confirm**).

## 1. Architecture

A single Python process with five boundaries:

```
agent/discovery ─┐                      ┌─ policy (allowlist, risk)      ┌─ handoff store (JSON files) ◄─ operator console / CLI
                 ├─► Surface protocol ◄─┤                                │
replay engine  ──┘   (web: Playwright)  └─ redaction ─► run log/evidence └─ Controller (automation side of control transfer)
recorder: trace ─► Capability  ·  AppProfile (per product version)  ·  TenantConfig (per institution)
```

**Key decisions:**

- **Perception is semantic, not structural.** A single JS library computes, for every visible control in every frame, a role, an *accessible name*, and an `nth`. The name is derived the way a screen reader or a human would read it: aria → `<label>` → button caption/link text → **caption in the neighbouring table cell** → placeholder. The *same* library feeds the LLM's observation, replay's target resolution, and the capture of a human operator's actions. What the model saw, what the artifact stores and what replay matches therefore cannot disagree.
- **The LLM sees text (frame text plus a control inventory with refs), not screenshots.**
  - This is cheap and deterministic, and refs map one-to-one onto recordable targets.
  - Screenshots are captured anyway, as evidence.
  - The trade-off: this approach needs a readable surface (DOM, or UIA on desktop). Pixel-only surfaces need a vision-grounding implementation of the same `Surface` protocol (§4).
- **A stateless, one-action-per-turn loop.** Each turn sends the goal, the parameters, a compact history and the *current* observation. Prompts stay bounded, each decision is independently logged, and providers are interchangeable (Anthropic `claude-opus-5` by default; OpenAI; a local model through Ollama; a scripted stand-in for tests). The evidence discovery runs used **Qwen 2.5 7B locally via Ollama**: no API key, and screen data never leaves the machine, which is also the posture a bank would want for discovery on real screens. The loop owns every safety decision; the model only proposes.
- **Why a single process:** replay is the production path, and it is a function `replay(capability, inputs, tenant) → ReplayResult`. The handoff store is file-based, so the operator side already runs out-of-process. A queue or worker pool can wrap this later without changing the contract.

## 2. Artifact schema

(`cua/schema.py`; see `evidence/capability.*.json` and the `.review.txt` renderings.)

Three separately versioned layers, because they change at different rates and are owned by different people:

| layer | granularity | contains | changes when |
|---|---|---|---|
| **Capability** `cua.capability/v1` | one task | contract + steps | the task changes |
| **AppProfile** `cua.app_profile/v1` | vendor product@version | sign-on flow, runtime-state taxonomy, PII captions | the vendor ships a version |
| **TenantConfig** | institution | base URL, secret *references*, caption overlays | a tenant configures differently |

A Capability is a **callable contract first and a step list second**:

- `id`, `version` (semver), `status` (`draft → approved → deprecated`), `title`/`description`, and `app` (the profile + version it targets).
- `inputs`: a typed `ParamSpec` per input (type, pattern, enum, min/max, **sensitivity**). Types are inferred from discovery. For example, a value fed into a `<select>` becomes an `enum` of the options actually on screen. A reviewer can tighten them before approval.
- `outputs`: type (`money` is normalised to a decimal string, never a float) and sensitivity.
- `outcomes`: the business outcomes a caller must be ready for (`member_not_found`, `access_restricted`, `invalid_input`). These are a declared part of the contract, not error strings.
- `steps[]`: each has `intent` (human-readable), `action` (a discriminated union: click/fill/select/press/navigate/extract), `risk` (`read`/`write`/`irreversible`), an `expect` checkpoint (postcondition), and `source` (`model`/`human`/`authored`).
- `success`: the final checkpoint. `provenance` records the discovery run id, model, goal template, human-step count and reviewer.

**Values** are `literal | param | secret` references. The artifact never contains discovery example values: routes, titles and texts are canonicalised (`/mbr/10042` → `/mbr/{member_id}`) and a unit test asserts this. It never contains secrets either, only names that the tenant maps to its own vault or env.

**Targets** are `(frame, role, name, nth)` plus an ordered list of *semantic* fallbacks (`input[name="q"]`, a button caption). Positional CSS (`nth-of-type` chains) is **deliberately not recorded**. When layout shifts, it silently clicks the wrong control, which is the worst possible failure for a bank. I prefer to fail loudly.

**Extraction** is semantic too: `labeled_value` (the value next to a caption), `table_cell` (row found by the *value* of a key column and column found by *header*, so reordered rows still work), and `text_regex` as a last resort.

Validation happens at load time: every `{param}` must be declared, every output needs exactly one extract step, and step ids must be unique. `python -m cua show` renders the artifact for review. `python -m cua catalog` renders approved capabilities as tool definitions for an agent.

## 3. Determinism & error handling

**Determinism.** Replay never consults a model. For each step (`cua/replay.py`), the engine:

1. asserts that automation holds control;
2. classifies the current screen;
3. applies the approval gate if the step is irreversible;
4. resolves the target by role + accessible name, polling until it appears;
5. acts;
6. **asserts the step's postcondition** (route + title of the frame that changed).

There are no sleeps; everything is a condition with a timeout. A screen is only classified once it is the *result* of the action: a navigation has committed and no document request is still pending, or it has persisted for 1.5 s. This came from a real bug: with a slow host, the engine was classifying the *previous* page.

**Error taxonomy.** The runtime states live in the app profile, because they belong to the product, not to one task. Each state has detectors and a classification:

| class | examples | engine behaviour | caller sees |
|---|---|---|---|
| business | no such member, restricted account, host validation `ERR 1102` | stop immediately, capture the host's own message | `status: business_outcome`, `outcome.code` |
| recoverable | maintenance interstitial, session expired / signed out, slow host | bounded recovery (click, wait, or re-run the sign-on flow then **restart**); counted per code | success, with a `recoveries[]` list |
| hard failure | host ABEND, credentials rejected | stop, never retry (the host may have partially processed the request) | `status: failed`, `failure.code: host_error` |
| unrecognised | a screen that is neither the checkpoint nor any known state | escalate to a human if enabled, else fail | `failure.code: checkpoint_timeout` |

Two guards shape the recoveries:

- A restart is refused once an irreversible step has run, because it could duplicate a transaction. The run fails with `recovery_exhausted`, and the evidence goes to a human.
- Requests are validated against `inputs` *before* the UI is touched, which yields `invalid_request`.

**Failures are debuggable.** Every failure carries the step id and intent, the *expected* condition, the *observed* frames (redacted path, title and text), a PII-masked screenshot, and a redacted DOM digest (frame text plus control inventory). `evidence/09-*` and `evidence/10-*` show examples.

**UI drift** (secondary here, since these UIs are stable). If a fallback locator is used, the result reports `drift[]` so the artifact can be re-baselined. Missing targets fail loudly, and ambiguous fallbacks are refused rather than guessed.

## 4. Heterogeneity & multi-tenant

**Surface abstraction.** The seam is `cua/surface/base.py`: `observe`, `resolve(Target)`, `click/fill/select/press`, `extract(source)`, `check(condition)`, `screenshot`, and `on_human_action`. Nothing above it knows about the DOM. The artifact's vocabulary (window/frame, role, accessible name, caption, table header) maps directly to other surfaces:

- **Legacy web:** implemented. Framesets, table captions and JS buttons are exactly the hard parts handled here. iframes and nested framesets are covered by frame names. Server-side session loss is covered by the profile's sign-on flow.
- **Desktop (Win32/WinForms/WPF/Java):** a `UIASurface` built on UI Automation. `frame` → top-level window plus pane path, `role` → ControlType, `name` → UIA Name (or the adjacent static label, the same heuristic as table captions), `css` fallbacks → AutomationId. Checkpoints become window title + focused pane.
- **Pixel-only (Citrix, green screen):** a `VisionSurface` would produce the same `Observation` from OCR plus a grounding model. Targets resolve by caption text near a control. For 3270/5250 terminals, a screen-scraping surface would use field positions and screen ids as checkpoints.

**Multi-tenant reuse.** The unit of reuse is the *vendor product version*, not the tenant:

- **Record once, per product version.** Capabilities target an `AppProfile@version` (`acme-coreteller@4.2`) and hold no tenant data: no URLs, no secrets, no example values. A tenant supplies `base_url`, secret references and a small **overlay**.
- **Caption aliases (implemented).** `evidence/12-replay-tenant-b` runs the unchanged tenant-A artifact against tenant B, which relabelled "Member Number" as "Acct/Member #" and "Balance" as "Current Bal", via `aliases` in `tenants/rvfcu.json`. Without the overlay the same run fails loudly with `target_not_found` (this is tested).
- **Further overlays (designed).** Step-level overrides (insert, skip or replace a step by id, for a tenant with an extra confirmation page) and profile known-state additions.
- **Resolution order:** capability → product-version profile → tenant overlay. Each tenant can pin versions and roll out new ones gradually.

**Detecting drift per tenant or version:**

- **Passive:** every replay reports fallbacks used (`drift[]`), unrecognised-state escalations and checkpoint timeouts, keyed by tenant and capability version. A rising rate on one tenant signals a local configuration change. A rising rate across all tenants on one product version signals a vendor upgrade.
- **Active:** canary replays of read-only capabilities on a schedule (the stretch-goal stability metric).
- **Versioning signals:** the profile can declare a version detector (e.g. a footer build string).
- **Remediation:** discovery re-runs against the changed tenant, and the diff against the base artifact becomes either a tenant overlay or a new capability version. Humans review both paths.

## 5. Escalation & handoff

**Detecting "stuck".**

- *Discovery:* step budget exhausted, the same action on an unchanged screen three times, three consecutive failed actions, an unverifiable `finish` claim, provider errors, or the model's own `request_help`.
- *Replay:* a screen that matches neither the checkpoint nor any known state (recoveries that run out, and restarts after an irreversible step, fail with evidence instead).
- *Both:* any irreversible action needs a human decision (an *approval* request).

**Routing.** An `InterventionRequest` (in `runs/<id>/handoff/`) carries:

- kind (`stuck` / `approval`), reason, and subject (capability ref or goal);
- step id and intent, and the proposed action;
- redacted frame state and a masked screenshot;
- a redacted DOM digest;
- the **allowed decisions** for that kind.

The operator console (`:5056`) or CLI lists open requests. In production this store is replaced by the existing work queue (ServiceNow or similar); the request is the payload.

**Control-transfer model.** There is exactly one controller per live session, recorded in `control.json` as `{holder: automation|human, operator, epoch}`, with every transfer appended to `control_log.jsonl`:

- Automation calls `ensure_automation()` before *every* action. It cannot act while a human holds control.
- **Claim:** the human takes control (holder = human, epoch + 1). Automation stays blocked but keeps the **same** browser session alive, and the human operates it directly.
- **Resolve:** the decision hands control back (epoch + 1).
- Epochs let any stale actor detect that it lost control.
- While a human holds control, their clicks, changes and selects are captured by the shared perception library as `role "name"` in `frame`. They are logged as `human_action` events and attached to the request. Password values are never read.
- **On resume, automation trusts nothing.** It re-observes and re-verifies the step's checkpoint. If the human completed the step, the run continues; if not, it fails with the expected/observed evidence.
- In discovery, the human's actions are appended to the trace and recorded into the artifact as `source: "human"` steps. The human's fix becomes part of the capability, visibly marked for review.

`evidence/11-replay-unknown-screen-handoff` shows the whole sequence: request → control to human (epoch 1) → human action captured → control to automation (epoch 2) → re-verification → success.

**Built vs. designed.** Built: pause, claim, live takeover (with `--headed`), capture, and resume. For demos and tests, `SimulatedOperator` claims the request and drives the live page with real input events. Designed only: a streamed co-browsing console (CDP screencast, or the browser in a VDI the operator connects to), plus assignment, SLAs and paging.

## 6. Safety

**Allowlist, at three layers:**

1. a static check of the artifact before replay (action types, forbidden controls, and whether risk labels agree with policy);
2. a per-action check in both loops;
3. **network interception**: every document navigation, including ones caused indirectly by a click, is checked against allowed origins and route globs (`/__faults`, `/logout` are denied) and aborted if outside.

A prompt-injected model therefore cannot reach anything the policy does not allow.

**Risk classes:**

- `read`: navigation, form filling (local state only).
- `write`: Continue/Save, i.e. reversible or draft actions.
- `irreversible`: Confirm/Submit/Transfer/Delete. **Every irreversible step requires per-invocation human approval** by default, in discovery and in replay. With nobody to approve, it is never executed (the run fails with `escalation_aborted`). A capability can be allowed to run irreversible steps unattended only if it is `approved` **and** listed in `unattended_irreversible`.
- *Forbidden* controls (Sign Off) and password fields are never touched by the model.

**Justification:** blocking outright would make write capabilities useless; flagging after the fact is too late for money movement.

**Secrets and PII:**

- Credentials exist only as secret *names* in artifacts. They are resolved at execution time from the tenant's provider (env here, a vault in production). The model never sees or types them: sign-on is a profile flow run before discovery.
- Logs, results, intervention requests and DOM digests all pass through a redactor. It masks known secret values, `pii`-classified inputs and outputs (keeping a two-character suffix for correlation), and patterns: SSN, Luhn-valid card numbers, emails, phone numbers.
- Screenshots mask password fields, values next to the profile's `pii_labels` captions, and SSN-shaped cells.
- The LLM prompt is pattern-redacted too.

**Limits, stated plainly:**

- Free-text PII (a name in an unlabelled paragraph) is only caught if it is a known value or sits next to a labelled caption.
- During discovery the model provider sees screen text, so production needs a provider agreement with zero data retention, plus synthetic or test members for discovery.
- Risk classification by caption is a heuristic. The profile or policy should explicitly mark commit controls per product.
- The file-based handoff store has no authentication; it would be replaced by the queue's identity and ACLs.

## 7. Cuts

**Cut deliberately:**

- **Desktop and vision surfaces:** designed (§4), not built.
- **A real co-browsing console:** the console is minimal, and live takeover uses the headed browser window.
- **Multi-tenant registry and rollout service, queues, workers:** the brief rewards the abstractions, not this infrastructure.
- **Step-level tenant overlays:** only caption aliases are implemented.
- **Bounded LLM "assisted fallback" in replay:** today a stuck replay goes to a human, never to a model.
- **Trace pruning:** the recorder keeps every successful action, so a model detour becomes a step. Reviewers see and can edit this.

**What a small local model taught the harness.** Discovery for the evidence ran on a 7B model on a laptop CPU. It completed the lookup flow on its own, but not the longer sub-account form (run `14`; see the README for how that capability was recorded instead). Its failure modes were useful, and each became a general guard rather than a prompt tweak for this app:

- **Grounding check:** every UI action must also name the control it means, and a mismatch with the ref is rejected with the correct ref, so a mis-click never happens silently. Name formatting is ignored when comparing.
- **Text-form tool calls:** a tool call written into the message text is accepted only if it names a known tool with a JSON-object argument. Anything else is sent back to the model with an instruction to call exactly one tool.
- **Unverifiable success text is discarded, not recorded:** an invented confirmation phrase would become a checkpoint that replay could never meet.
- **Sensitivity is decided by policy, not the model:** the model labelled a member's name `internal`. Values under the app profile's PII captions are now forced to `pii`.
- **Symbolic parameters:** the model may type `{member_id}`, which the harness resolves to the value. It once typed the parameter's name literally, and the host rejected it.
- **Stuck detection and handoff carry weak models:** loops and repeated failures are caught within three steps, and a human can steer once and hand back.

**Built beyond the core:** two optional stretch goals. One is the agent-facing catalog / `invoke`. The other is approval gating (`draft → approved`), with `unattended_irreversible` for irreversible steps.

**Next, in order:**

1. A stability score from N replays and nightly canaries, gating `approved` status (confidence).
2. An LLM-assisted *proposal* of new known states from escalations: the human's `note` plus the captured actions become a draft profile entry, the loop the survey demo points at.
3. Step-level overlays and profile version detection.
4. A `UIASurface` for one real desktop app, to prove the seam.
5. Trace minimisation before recording (dropping no-op detours via replay-and-diff).
