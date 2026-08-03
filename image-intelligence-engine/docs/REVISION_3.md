# Revision 3 — Custody, Lifecycle and Reproducibility Refinements

Amendment to [ARCHITECTURE.md](ARCHITECTURE.md) and
[DATA_MODEL.md](DATA_MODEL.md), agreed before Phase 2 implementation.

Everything in R2 is retained: API + Worker topology, PostgreSQL graph model,
evidence-first principle, Observation → Mention → Entity → Fact, no biometric
processing, provider capability interfaces, read-only copilot, immutable audit
log, local-first Docker deployment.

**Unifying theme.** R2 made evidence *traceable*. R3 makes it *defensible over
time*: who handled it, what state it is in, why it still exists, and whether a
third party can reproduce the finding without access to this server.

---

## 1. Evidence lifecycle management

### 1.1 The states are two axes, not one chain

A naive reading makes these sequential. They are not. `DISCOVERED → DOWNLOADED →
VERIFIED → REVIEWED → INCLUDED_IN_REPORT` describes **investigative progress**,
while `ARCHIVED / RETAINED / PURGED` describes **retention disposition**. An
artifact can be `INCLUDED_IN_REPORT` *and* under a retention hold at once.
Collapsing them into one column would make "included in a report but pending
deletion" unrepresentable — which is precisely the state that matters most for
compliance.

So: two columns, each a small state machine.

```
progress_state:   DISCOVERED → DOWNLOADED → VERIFIED → REVIEWED → INCLUDED_IN_REPORT
retention_state:  ACTIVE → ARCHIVED → RETAINED → PURGED        (PURGED is terminal)
```

### 1.2 Transitions are validated, not assigned

Allowed transitions live in the **domain layer** as a pure table. A repository
refuses an illegal transition, and every accepted transition writes an
`evidence_lifecycle_events` row. Nothing sets a state by direct assignment.

**Deletion protection:** `PURGED` is reachable only when no active retention hold
exists (§6) and the artifact is not referenced by a report that is itself still
retained. Purge is a *state transition subject to preconditions*, never a
`DELETE`.

```sql
-- added to images, pages, and any future artifact table
progress_state   text NOT NULL DEFAULT 'DISCOVERED',
retention_state  text NOT NULL DEFAULT 'ACTIVE',

evidence_lifecycle_events (
  id                uuid PRIMARY KEY,
  investigation_id  uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  artifact_type     text NOT NULL,          -- IMAGE|PAGE|SCREENSHOT|HTML_SNAPSHOT|REPORT
  artifact_id       uuid NOT NULL,
  axis              text NOT NULL,          -- PROGRESS|RETENTION
  from_state        text NOT NULL,
  to_state          text NOT NULL,
  reason            text NOT NULL DEFAULT '',
  actor_id          uuid REFERENCES users(id),   -- null = system
  occurred_at       timestamptz NOT NULL DEFAULT now()
)
```

---

## 2. Chain of custody

Distinct from the audit log, and worth stating why both exist. The **audit log**
answers "what did this system do?" — it is investigation-scoped and
action-oriented. The **custody chain** answers "what happened to *this
artifact*?" — it is artifact-scoped and follows one object through every
transformation.

An examiner challenging a screenshot asks the second question. Reconstructing it
by filtering the audit log would be possible but fragile; a first-class custody
record makes it a single query.

```sql
custody_events (
  id                uuid PRIMARY KEY,
  investigation_id  uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  artifact_type     text NOT NULL,
  artifact_id       uuid NOT NULL,
  sequence          integer NOT NULL,       -- 1-based, per artifact
  action            text NOT NULL,          -- COLLECTED|HASHED|TRANSFORMED|
                                            -- SCREENSHOT_CAPTURED|EXPORTED|
                                            -- INCLUDED_IN_REPORT|MIGRATED
  actor_id          uuid REFERENCES users(id),
  actor_kind        text NOT NULL,          -- HUMAN|SYSTEM
  source_uri        text NOT NULL DEFAULT '',
  content_hash      char(64) NOT NULL,      -- SHA256 at this point in time
  storage_location  text NOT NULL DEFAULT '',
  transformation    jsonb NOT NULL DEFAULT '{}',  -- tool, version, parameters
  derived_from_id   uuid REFERENCES custody_events(id),
  previous_hash     char(64) NOT NULL,      -- per-artifact chain
  entry_hash        char(64) NOT NULL,
  occurred_at       timestamptz NOT NULL DEFAULT now(),
  UNIQUE (artifact_type, artifact_id, sequence)
)
REVOKE UPDATE, DELETE ON custody_events FROM application_role;
```

**Every transformation creates a new record; nothing is overwritten.**
`derived_from_id` makes derivation explicit, so `image downloaded → hash created
→ screenshot captured → report generated` is a walkable chain rather than four
unrelated rows. Hash-chained per artifact and insert-only by privilege, matching
the audit log's guarantees.

---

## 3. Investigation status workflow

Replaces R2's `DRAFT|RUNNING|PAUSED|COMPLETE|ARCHIVED`.

```
NEW → ACTIVE → UNDER_REVIEW → COMPLETED → ARCHIVED → DELETED_PENDING_RETENTION
```

Transitions are validated in the domain layer. Notable rules:

- `ACTIVE → UNDER_REVIEW` requires at least one completed pipeline run.
- `UNDER_REVIEW → COMPLETED` requires the human review queue (§4) to be empty of
  `PENDING` items — a case cannot be completed with unreviewed machine output
  sitting in it.
- `DELETED_PENDING_RETENTION` is **not** deletion. It marks intent; the retention
  engine performs the purge only when policy allows and no hold exists, and the
  investigation row itself survives as a tombstone carrying its audit log.
- Backward transitions (`COMPLETED → ACTIVE`) are permitted but always audited
  with a mandatory reason. Real investigations reopen.

Operational status (`pipeline_runs.status`) stays separate from workflow status.
A case can be `UNDER_REVIEW` while a monitor sweep runs.

---

## 4. Human review queue

The layer that separates **machine observation** from **human-confirmed
interpretation**.

Extraction never writes a confirmed entity. It writes an observation plus a
review item. A human decision promotes it.

```sql
review_items (
  id                uuid PRIMARY KEY,
  investigation_id  uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  kind              text NOT NULL,   -- ENTITY_CANDIDATE|FACT_CANDIDATE|
                                     -- DUPLICATE_MERGE|CONFLICT|PROVENANCE_CLASS
  subject_type      text NOT NULL,
  subject_id        uuid NOT NULL,
  proposal          jsonb NOT NULL,  -- what the machine suggests
  rationale         jsonb NOT NULL,  -- why, incl. supporting observation ids
  priority          integer NOT NULL DEFAULT 0,
  status            text NOT NULL DEFAULT 'PENDING',  -- PENDING|CONFIRMED|
                                                      -- REJECTED|DEFERRED
  decided_by        uuid REFERENCES users(id),
  decided_at        timestamptz,
  decision_note     text NOT NULL DEFAULT '',
  created_at        timestamptz NOT NULL DEFAULT now()
)
CREATE INDEX ON review_items (investigation_id, status, priority DESC);
```

Entities gain `verification_state ∈ {MACHINE_PROPOSED, HUMAN_CONFIRMED,
HUMAN_REJECTED}`, defaulting to `MACHINE_PROPOSED`.

**A rejection never deletes the observation.** It records that a human declined
the interpretation. The underlying extraction remains, because "the machine saw
this and a human disagreed" is itself a finding worth preserving.

---

## 5. Finding classification

Two independent axes, and conflating them would be a real modelling error:

| Axis | Column | Values | Authored by |
|---|---|---|---|
| Evidential | `facts.status` | `COMMON` / `UNIQUE` / `CONFLICTED` / `UNKNOWN` | machine, from evidence |
| Investigative | `facts.classification` | `VERIFIED` / `UNVERIFIED` / `DISPUTED` / `REJECTED` / `NEEDS_REVIEW` | human judgement |

A fact can legitimately be `COMMON` (three independent sources agree) *and*
`DISPUTED` (the investigator has off-platform reason to doubt all three). One
column could not express that, and forcing it would silently overwrite either the
evidence or the judgement.

```sql
-- added to facts
classification        text NOT NULL DEFAULT 'UNVERIFIED',
classified_by         uuid REFERENCES users(id),
classified_at         timestamptz,
classification_note   text NOT NULL DEFAULT '',
```

**Rejected findings are never deleted.** They are excluded from report bodies,
retained in the evidence appendix, and remain queryable — the history of what was
considered and set aside is part of the investigation.

---

## 6. Data retention engine

```sql
retention_policies (
  id                uuid PRIMARY KEY,
  owner_id          uuid NOT NULL REFERENCES users(id),
  name              text NOT NULL,
  jurisdiction      text NOT NULL DEFAULT 'IN',
  default_days      integer NOT NULL,
  applies_to        text NOT NULL DEFAULT 'INVESTIGATION',
  auto_purge        boolean NOT NULL DEFAULT true,
  export_before_purge boolean NOT NULL DEFAULT true,
  created_at        timestamptz NOT NULL DEFAULT now()
)

retention_holds (
  id                uuid PRIMARY KEY,
  investigation_id  uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  artifact_type     text NOT NULL DEFAULT 'INVESTIGATION',
  artifact_id       uuid,
  reason            text NOT NULL,
  placed_by         uuid NOT NULL REFERENCES users(id),
  placed_at         timestamptz NOT NULL DEFAULT now(),
  released_by       uuid REFERENCES users(id),
  released_at       timestamptz
)
```

Engine rules:

1. **Automatic expiry** — scheduled job scans `retention_expires_at`.
2. **Manual override** — extending is permitted and audited with a reason.
3. **Preservation lock** — an unreleased `retention_holds` row blocks purge
   absolutely. Holds win over policy, always; a legal hold that a scheduler could
   override is not a hold.
4. **Export before deletion** — when `export_before_purge` is set, the
   reproducibility package (§7) is generated and its location recorded in the
   audit log *before* any artifact reaches `PURGED`.
5. **Tombstones survive** — purge removes artifacts and their content, never the
   investigation row or its audit log. Erasing the record of what was deleted
   would defeat the purpose of having a retention system.

---

## 7. Reproducibility package

A completed investigation exports as a portable, self-verifying archive.

```
case_export_<case_id>_<timestamp>.zip
├── manifest.json           # every file with SHA256, plus package hash
├── investigation.json      # case metadata, workflow history
├── config_snapshot.json    # §9 — versions used at each run
├── database/
│   └── investigation.sqlite   # portable relational snapshot
├── images/
├── screenshots/
├── html/
├── timeline.json
├── graph.json
├── facts.json              # incl. evidence chains and confidence explanations
├── custody.jsonl           # §2, per-artifact chains
├── audit.jsonl             # hash-chained
├── reports/
└── VERIFY.md               # how to check integrity without this server
```

Two decisions worth recording. The database snapshot is **SQLite, not a pg_dump**
— a reviewer with Python and no PostgreSQL install can still open it, and
portability is the entire point. And `manifest.json` carries a hash for every
file plus a hash over the manifest itself, so tampering is detectable offline.

`VERIFY.md` documents re-walking the audit and custody chains and re-checking
file hashes, so verification needs no proprietary tooling.

---

## 8. Search provenance

Every discovery call is recorded — how evidence was found is itself evidence.

```sql
discovery_requests (
  id                uuid PRIMARY KEY,
  investigation_id  uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  pipeline_run_id   uuid REFERENCES pipeline_runs(id),
  provider          text NOT NULL,
  provider_version  text NOT NULL DEFAULT '',
  capability        text NOT NULL,
  probe_image_id    uuid REFERENCES images(id),
  query_parameters  jsonb NOT NULL DEFAULT '{}',
  requested_at      timestamptz NOT NULL DEFAULT now(),
  responded_at      timestamptz,
  http_status       integer,
  raw_response_key  text NOT NULL DEFAULT '',   -- object store
  results_returned  integer NOT NULL DEFAULT 0,
  results_accepted  integer NOT NULL DEFAULT 0,
  results_rejected  integer NOT NULL DEFAULT 0,
  rejection_reasons jsonb NOT NULL DEFAULT '{}',
  error             text NOT NULL DEFAULT '',
  cost_units        double precision
)
```

Retaining the raw provider response matters: when a provenance classification is
challenged, the question is often "what did the provider actually return?" rather
than "what did we conclude?". `results_rejected` with reasons makes local
verification (ARCHITECTURE §8) auditable — we can show exactly which provider
claims were discarded and why. `cost_units` supports the per-investigation
budget cap in the risk register.

---

## 9. Configuration versioning

An investigation remembers the system that produced it, or reruns are not
explainable.

```sql
config_snapshots (
  id                uuid PRIMARY KEY,
  investigation_id  uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  pipeline_run_id   uuid REFERENCES pipeline_runs(id),
  app_version       text NOT NULL,
  ruleset_version   text NOT NULL,
  parser_versions   jsonb NOT NULL DEFAULT '{}',   -- extract, extruct, simhash…
  extractor_versions jsonb NOT NULL DEFAULT '{}',  -- spaCy model, OCR engine
  provider_versions jsonb NOT NULL DEFAULT '{}',
  prompt_versions   jsonb NOT NULL DEFAULT '{}',   -- copilot, adjudication
  scorer_version    text NOT NULL,
  classifier_version text NOT NULL,
  thresholds        jsonb NOT NULL DEFAULT '{}',   -- provenance hamming bounds
  captured_at       timestamptz NOT NULL DEFAULT now()
)
```

Captured at the **start of every pipeline run**, not once per investigation — a
case re-run after a spaCy upgrade has two different extraction regimes in it, and
the report must be able to say which findings came from which.

Existing `extractor_version` / `scorer_version` / `classifier_version` columns on
individual rows stay: the snapshot gives the run-level picture, the row-level
column gives per-artifact precision.

---

## 10. Testing fixtures

Controlled datasets, built **before** Phases 8 and 9 rather than alongside them,
so algorithm changes are measured against an unchanging yardstick.

```
tests/fixtures/
├── MANIFEST.yaml            # every case: inputs, expected outputs, provenance
├── images/
│   ├── duplicates/          # exact, resized, compressed, mirrored, cropped
│   └── unrelated/           # true negatives — the set most often forgotten
├── pages/
│   ├── syndication/         # one wire story across many domains
│   ├── conflicting/         # disagreeing employer / location
│   ├── structured/          # schema.org Person, microdata, RDFa
│   └── malformed/           # broken markup, wrong encodings
├── changes/                 # before/after page pairs for monitoring
└── archive/                 # wayback response shapes incl. gaps
```

`MANIFEST.yaml` declares expected outcomes per fixture, so Phase 8 and 9
acceptance is a measured number rather than an opinion. Fixture images must be
synthetic or licensed — **no scraped photographs of real people**, which would
reintroduce the exact collection problem the architecture exists to avoid.

Every algorithm change reports its score against the same fixture set, and the
measurement is recorded in the docs alongside the threshold it justifies.

---

## 11. Impact on module map and roadmap

Two new modules:

```
retention/     # policies, holds, expiry scanner, purge with preconditions
export/        # reproducibility package builder + verifier
```

`custody/` is deliberately **not** a module — custody recording is a
cross-cutting concern belonging in `shared/` alongside the audit writer, because
every module that touches an artifact must be able to record custody without
importing a sibling.

Roadmap changes:

| Phase | Change |
|---|---|
| 2 | Adds lifecycle/status/custody enums and transition tables to `shared/`; CI guards unchanged |
| 3 | Adds `custody_events`, `evidence_lifecycle_events`, `review_items`, `retention_policies`, `retention_holds`, `discovery_requests`, `config_snapshots` |
| 4 | Adds investigation workflow state machine, retention hold API |
| 7 | Every provider call writes a `discovery_requests` row |
| 8, 9 | **Gated on fixtures existing first** (§10) |
| 10 | Extraction emits `review_items` rather than confirmed entities |
| 14 | Reports honour `classification`; rejected findings appear in the appendix only |
| 15 | Adds retention engine and export package to acceptance |

New phase, slotted after Phase 12:

**Phase 12b — Retention and reproducibility.** Retention engine with holds and
export-before-purge, plus the reproducibility package and its offline verifier.
Done when a purge is blocked by a hold, an export verifies offline with no
server access, and a tombstoned investigation retains its audit log.
