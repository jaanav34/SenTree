# Hephaestus Canonical Operating Contract

This is the canonical operating contract for Hephaestus and is intended to be inherited or adapted by target repositories.

## Install Assumptions

- Official install/distribution path is the Python CLI package command surface (`hephaestus ...`).
- Compatibility wrappers (`scripts/install-local.*`, `scripts/bootstrap-target-repo.sh`) remain temporarily and must delegate to CLI with deprecation notice.
- Target repositories are expected to have `.codex/`, `.agents/skills/`, and `.hephaestus/` paths available after install.
- This contract is the source-of-truth policy. Target repositories may add stricter constraints but should not weaken these baseline gates.

## Normal User Mode

- Default interaction mode is normal prompting (`hephaestus ask "<natural prompt>"`).
- Internal role orchestration and lifecycle enforcement remain mandatory but are hidden by default.
- User-facing output should emphasize milestones, blockers, ship status, and next command.
- Advanced/operator controls may expose explicit stage routing, but they must not relax mandatory gates.

## Mandatory Lifecycle

All work follows this exact progression:

1. Request
2. Intake
3. Spec
4. Plan
5. Execute
6. Verify
7. Review
8. Ship
9. Retro

Rules:

- No stage skipping.
- No stage may be marked complete without its required evidence in `.hephaestus/` and/or the DB, according to task-size policy.
- Implementation must be spec-linked. If code work is requested without a spec, route back to Spec.
- Verification is required before completion.
- Review is required before shipping.
- OWASP security audit is required before shipping (as a mandatory review sub-gate).
- `T0/T1` work may satisfy lifecycle evidence with compact artifacts; `T2/T3` keep the fuller artifact model.

## Hephaestus Code Quality Constitution

These are enforceable rules. Violations are blocking defects and must be corrected before Ship.

Before introducing a new pattern, helper, service, abstraction, build step, test style, or configuration shape, first search the repo for the closest existing equivalent and prefer extending or matching that pattern.

### Non-Negotiable Prohibitions

- Code MUST NOT hardcode deploy-varying config, secrets, ports, URLs, environment names, credentials, or branch names.
- Code MUST NOT introduce magic numbers or magic strings without named constants, config, enums, or explicit domain rationale.
- Code MUST NOT create bloated files, giant classes/components, or long methods when decomposition is warranted.
- Code MUST NOT duplicate logic that should be extracted into shared units.
- Code MUST NOT use long parameter lists, flag arguments, or mode-switch booleans when a clearer abstraction is needed.
- Code MUST NOT implement only happy-path behavior while ignoring failures, edge cases, rollback, empty states, partial data, or error propagation.
- No role MAY declare work complete without real verification evidence.
- Code MUST NOT swallow errors or emit vague logging that prevents triage.
- Changes MUST NOT introduce heavyweight dependencies when existing repo tools or simpler code can satisfy the need.
- Changes MUST NOT drift from established repository patterns without explicit written justification.
- Changes MUST NOT include placeholder comments, TODO dumping, or scaffolding without immediate execution value.
- No significant architecture/interface change MAY proceed without updating spec/plan architecture notes.

### Required Quality Rules

- Deploy-varying configuration MUST be externalized and validated at startup.
- New logic MUST be placed near existing patterns unless a strong reason is documented in spec/plan artifacts.
- Each implementation MUST choose the smallest clean change that satisfies the approved spec.
- Code MUST be split by responsibility and optimized for future modification, not just immediate completion.
- Every changed behavior MUST map to acceptance criteria and verification evidence.
- All failures MUST produce actionable logs and corresponding DB-backed evidence or compact local cache records.
- Any change touching CI, build, deploy, or runtime configuration MUST involve `devops_infra`.
- Any change touching architecture or interfaces MUST be reviewed by `system_architect`.
- `implementer` MUST NEVER self-approve completion.
- Code writing alone MUST NEVER be treated as completion.
- Passing one narrow test MUST NEVER be treated as sufficient if changed behavior has broader impact.
- Verification MUST be derived from both acceptance criteria and changed surface in the diff.
- Verification SHOULD include at least one regression-oriented check for changed behavior when feasible.
- Failures MUST be recorded in reports/logs and MUST NOT be hidden.
- If tests are absent, that absence MUST be explicitly recorded as a risk.
- `reviewer` MAY block ship solely for insufficient verification evidence.

### Enforcement

- `qa_engineer` blocks incomplete verification, weak failure handling, and non-reproducible behavior.
- `reviewer` blocks correctness, security, maintainability, and anti-pattern violations.
- `scrum_master` blocks stage completion when required evidence, role reviews, or handoffs are missing.
- `devops_infra` blocks Ship for unresolved CI/build/deploy/runtime risk.
- `system_architect` blocks Ship for architecture/interface drift without approved notes.

## OWASP Top 10:2025 Security Enforcement

OWASP Top 10:2025 is a mandatory planning, execution, verification, and review lens.

The following categories are first-class and must be explicitly checked:

- `A01 Broken Access Control`
- `A02 Security Misconfiguration`
- `A03 Software Supply Chain Failures`
- `A04 Cryptographic Failures`
- `A05 Injection`
- `A06 Insecure Design`
- `A07 Authentication Failures`
- `A08 Software or Data Integrity Failures`
- `A09 Security Logging and Alerting Failures`
- `A10 Mishandling of Exceptional Conditions`

Rules:

- Spec/plan artifacts must identify relevant OWASP categories for the changed surface.
- Verify/review/laziness-audit must produce OWASP-tagged findings and evidence when risk exists.
- Material security findings must be persisted to DB in `security_findings`, mirrored in `review_findings`, and linked to `blockers` when blocking.
- Security retrieval evidence must use security chunk types in `retrieval_chunks` when security write-back is required.
- For low-risk `T0/T1` work, one compact OWASP summary is sufficient unless the changed surface touches security-sensitive domains or yields a real finding.
- Raw evidence is fallback-only for high-risk or resumed work; compact local tasks may use raw/code-first inspection when prior memory is not useful and then backfill only the minimum required evidence.

Default ship-blocker policy:

- Block by default: `A01`, `A02`, `A03`, `A04`, `A05`, `A06`, `A07`, `A08`.
- Conditional by default (reviewer may escalate to block): `A09`, `A10`.
- Reviewer may always block ship when security evidence is insufficient.

## Decomposition and Size Discipline

These are mandatory design and delivery constraints.

- Prefer extending an existing small module over creating a giant new one.
- Do not stuff an entire feature into one file when responsibilities naturally split.
- If a file grows substantially, include a brief note explaining why extraction was not appropriate.
- Long methods/functions/components must be decomposed aggressively.
- Phase plans must prefer small reviewable increments.
- If a change becomes broad, split it into multiple phases or multiple reports.
- Each implementation report must state whether file/class/function growth stayed reasonable.

## Artifact Contract

Hephaestus artifacts are mandatory delivery evidence.

- `T0 micro`
  - Primary evidence: task-service `evidence` rows plus compact task-context summary in DB.
  - Request, acceptance criteria, design note, verification, review, and OWASP notes must be stored as structured DB evidence metadata.
- `T1 small`
  - Primary evidence: task-service `evidence`, `ci_runs`, and `review-comments`.
  - Dedicated verification/review markdown is forbidden for runtime flow; use compact DB summaries and references only.
- `T2/T3`
  - Request, intake, spec, plan, execute, verify, review, ship, and retro states must be represented as DB-backed lifecycle evidence.
  - Raw logs or full transcripts may exist only in the local ephemeral cache DB and must not be promoted to runtime markdown artifacts.
- Ship decisions are captured as DB evidence plus blocking/non-blocking review records.
- Retros are DB-only runtime records and should be compact unless a non-runtime document is explicitly requested.

## DB-First Memory Contract

Hephaestus V2 uses a local DB-first memory layer in addition to files.

- Default local DB path: `.hephaestus/db/hephaestus_memory.sqlite3`.
- Migration source: `.hephaestus/db/migrations/`.
- Migrations are applied with `scripts/db-migrate.sh` or `scripts/db-migrate.ps1`.
- Retrieval policy references:
  - `.hephaestus/prompts/db-navigation-contract.md`
  - `.hephaestus/prompts/minimal-task-shapes.md`
- Standard API path: `.hephaestus/db/storage/retrieval_api.py` (skills/roles should use this, not ad hoc scans).
- DB navigation contract: `.hephaestus/prompts/db-navigation-contract.md`.
- DB-first is mandatory for resumed work, multi-agent coordination, CI-heavy investigation, and `T2/T3` tasks.
- Fresh local `T0/T1` work may inspect code/artifacts first when prior DB memory is low-value, then backfill concise evidence only if needed for blockers, findings, handoffs, or resumption.
- Every role SHOULD query intent/contract/risk summaries from DB before broad raw-code exploration when useful summaries exist.
- Every stage MUST write the minimum DB evidence needed for coordination; `artifact_index`, `handoffs`, `blockers`, and `agent_runs` are required when artifacts, handoffs, blockers, or resumability matter.
- Raw files remain the source fallback evidence, but only metadata and paths are stored in DB (`raw_artifacts`).
- If DB context is incomplete, roles may open raw artifacts and then backfill concise retrieval records (`retrieval_chunks`, `facts`, `verification_results`, `review_findings`).
- Raw code fallback is required to follow persisted summary precheck (`intent -> contract -> risk`) for high-risk or resumed work; `T0/T1` local tasks may use a lighter audited precheck when there is no meaningful prior memory.
- Recurrence intelligence MUST be used for repeated failures (`error_signatures`, `recurrence_history`, `recurrence_patterns`) before proposing net-new root causes.

## Product Planning Memory Contract

Product planning is an additive hidden path inside `hephaestus ask` and remains DB-first.

- Planning states are DB-driven and resumable:
  - `intake`
  - `idea_expansion`
  - `feature_extraction`
  - `deduplication`
  - `feature_refinement`
  - `requirements_definition`
  - `task_breakdown`
  - `roadmap_shaping`
  - `validation`
  - `ready_for_output`
- Every planning state transition MUST persist:
  - evidence references (`retrieval_chunks`, `facts`, `referenced_files`, `file_extracts`)
  - decision trace (`feature_decisions`, `decision_rationale`)
  - confidence and maturity updates on the active planning thread.
- No planning output files are written before readiness gate passes.
- Readiness requires deduped features, constraints, requirements, tasks, and open-question coverage with confidence threshold.

## Planning Dedup Contract

- Feature normalization MUST include: `problem`, `user_value`, `intent_tags`, `dependencies`, `constraints`, `risk_tags`.
- Dedup scoring MUST consider semantic similarity + intent overlap + dependency overlap + user-value overlap.
- Default merge policy is `balanced_auto_merge`:
  - auto-merge at high confidence (`>= 0.86`)
  - suggest merge at medium confidence (`0.72..0.85`)
  - keep separate below threshold.
- Every merge/suggestion decision MUST persist score, source features, rationale, and resulting feature linkage.

## Agent Collaboration Contract

- Planning uses two-pass peer sync:
  - primary specialist output
  - peer critique
  - decision recorder finalization.
- Critique and final rationale MUST be persisted in `decision_rationale`.
- `decision_recorder` is the final authority for planning decision persistence.

## Direction Suggestion Contract

- Direction suggestions are post-stabilization only (after validation/readiness checks).
- Suggestions MUST be grounded in persisted planning memory (no chat-history-only reasoning).
- Suggestions MUST persist as decision records (`direction_suggestion`) with evidence references.

## Delegation Model

The orchestrator delegates by specialist role with explicit handoffs.

- Read-heavy work can run in parallel subagents.
- Write-heavy work is serialized to one active writer at a time.
- Implementer is never the final approver of completion.
- Specialist outputs must reference source artifacts and the next owning role.

## Verification Memory Rule

If verification commands are unknown:

1. Ask once for the project verification commands.
2. Persist the answer to `repos.settings.hephaestus.verification`.
3. Reuse persisted commands in future runs unless the user changes them.

## Quality Feedback Loop

If repeated corrections occur for the same failure pattern:

- Update the Hephaestus system itself, not only the current feature branch.
- Record the corrective rule as DB evidence in the retro stage.
- Propose updates to prompts, runbooks, role contracts, or templates.

## Local vs Worktree Mode

Local mode:

- Usually copy install.
- Artifacts and logs are written directly in the target repository.
- Best for normal repo adoption and team commits.

Worktree mode:

- Often symlink install during Hephaestus development.
- `scripts/setup-hephaestus.sh` should run during environment setup to wire and verify paths.
- Keep write operations serialized in the active worktree to avoid cross-worktree drift.

## Target Repository Responsibilities

- Maintain valid `AGENTS.md` and `.codex/config.toml` with Hephaestus managed policy blocks.
- Keep runtime lifecycle state in database-backed evidence; checked-in `.hephaestus/` markdown artifacts are not part of the runtime contract.
- Provide or confirm verification commands when first requested.
- Run Verify and Review gates before any Ship decision.
- Capture Retro outcomes and fold repeated fixes into system improvements.

## Logs and Reports Storage

- Operational logs belong in the local ephemeral DB/cache unless a non-runtime export is explicitly requested.
- Validation, review, shipping, and handoff evidence belong in DB-backed task/evidence/CI/review records.
- Runtime stage memory is database-backed and should be linkable from PRs/issues through task-service/Supabase records when possible.

## Orchestrator Role Contract

The orchestrator owns lifecycle enforcement and stage transitions.

- It routes work to intake, system_architect, scrum_master, implementer, qa_engineer, devops_infra, and reviewer roles.
- It enforces blocking rules from `.hephaestus/prompts/role-handoffs.md`.
- It rejects completion when required evidence or gate approvals are missing.

## Retrieval VNext Contract

These rules are additive to DB-first retrieval and are mandatory when retrieval vNext flags are enabled.

- Retrieval starts with intent classification and scope resolution (`run_local -> slug_local -> repo_local -> org_wide`).
- `org_wide` scope is denied by default unless explicitly enabled in config.
- Each retrieval call must persist an audit record in `retrieval_audit_log`.
- Evidence selection must respect role token budgets from `.codex/config.toml`.
- Raw fallback is only allowed after chunk-first insufficiency and must include persisted fallback reason.
- If retrieval produces no usable evidence, create a blocker rather than silently continuing.

## Long-Running Execution Loop Contract

- Stage progression must be reflected in `execution_checkpoints`.
- Ship readiness must include explicit `review_gates` and `security_gate_decisions`.
- Rework must be tracked with `repair_cycles` and `repair_attempts`; no untracked retry loops.
- Retry budgets must be enforced via `retry_budgets`; exhaustion escalates automatically.
- Idle/abandoned runs must be captured in `abandonment_events` and routed to `scrum_master`.

## CI Intelligence Contract

- CI triage must write `ci_triage_assessments` and `ci_verification_routes`.
- Flake inference must write `ci_flake_profiles` and may not auto-close blockers.
- Cluster/fix reuse should use `ci_failure_clusters` and `ci_fix_suggestions` before net-new guessing.
- Closure decisions must be explicit in `ci_closure_decisions` with residual risk and confidence.
- Do-not-auto-fix policy is mandatory for security-sensitive or low-confidence classes.
- Agents must read normalized CI DB state/chunks before opening raw CI pages or full raw logs.
- Hosted and local verification ingestion must normalize into DB (`ci_ingestion_batches`, `ci_ingestion_items`, `ci_log_fingerprints`) before triage/review.

## Deterministic Security Controls Contract

- OWASP findings remain required; deterministic controls are additive and must be stored in `security_controls` for high-risk surfaces, explicit findings, resumed work, or cross-agent release decisions.
- Security gate decisions must be persisted in `security_gate_decisions` before ship when a dedicated ship/security decision record is required.
- High/critical decisions must include linked evidence via `security_evidence_links`.
- Waivers must be explicit, time-bounded, and stored in `security_waivers`.

## Release Gate Contract

Release and upgrade flows must enforce hard gates in this order:

1. Schema compatibility (migrations apply, required tables/indexes exist).
2. Retrieval contract compatibility (core skills and roles reference DB navigation contract).
3. Security contract compatibility (OWASP/security contract references and security skill wiring are present).
4. Package compatibility (skills/templates/profile compatibility checks pass).
5. Self-test and smoke tests pass (`verify-install`, DB health full, targeted test suite).

Rules:

- A release fails if schema and runtime code are incompatible.
- A release fails if critical skills are missing DB navigation contract wiring.
- A release fails if security audit integration is missing.
- A release fails if self-test or smoke tests fail.

## Rollout Phases

New subsystems must support staged activation:

- `observe`: collect audit and metrics only.
- `warn`: emit blockers/warnings without full enforcement.
- `enforce`: hard-gate behavior and release decisions.

Default rollouts for additive subsystems should begin at `observe` unless security risk requires `warn`.
