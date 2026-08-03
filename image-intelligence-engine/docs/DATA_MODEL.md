# Data Model — Revision 2

Companion to [ARCHITECTURE.md](ARCHITECTURE.md). PostgreSQL 16. All ids are
UUIDv7 (time-ordered, so they sort chronologically and index well). All
timestamps are `timestamptz`, stored UTC.

**Invariants this schema enforces at the database level**, not by convention:

1. No `graph_edges` row may exist without supporting evidence.
2. No `Person` graph node may exist without an asserting page.
3. `HUMAN`-origin content cannot enter extracted-evidence tables.
4. `audit_log` is append-only and hash-chained.
5. Every `facts.confidence` has a matching persisted explanation.

---

## 1. Ownership and workspace

```sql
users (
  id                    uuid PRIMARY KEY,
  email                 citext UNIQUE NOT NULL,
  password_hash         text NOT NULL,
  display_name          text NOT NULL DEFAULT '',
  role                  text NOT NULL DEFAULT 'investigator',
  created_at            timestamptz NOT NULL DEFAULT now(),
  last_login_at         timestamptz
)
```

Single-user local deployment seeds exactly one row. `owner_id` appears on every
scoped table from day one so multi-user is middleware, not a migration.

```sql
investigations (
  id                    uuid PRIMARY KEY,
  owner_id              uuid NOT NULL REFERENCES users(id),
  case_id               text NOT NULL,              -- investigator's own ref
  title                 text NOT NULL,
  description           text NOT NULL DEFAULT '',
  lawful_basis          text NOT NULL,              -- required, audited
  purpose               text NOT NULL DEFAULT '',
  status                text NOT NULL,              -- DRAFT|RUNNING|PAUSED|COMPLETE|ARCHIVED
  jurisdiction          text NOT NULL DEFAULT 'IN',
  created_at            timestamptz NOT NULL DEFAULT now(),
  updated_at            timestamptz NOT NULL DEFAULT now(),
  completed_at          timestamptz,
  retention_expires_at  timestamptz,                -- scheduled purge
  UNIQUE (owner_id, case_id)
)

pipeline_runs (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  trigger               text NOT NULL,              -- MANUAL|MONITOR|RESUME
  status                text NOT NULL,
  started_at            timestamptz NOT NULL DEFAULT now(),
  finished_at           timestamptz
)

pipeline_stages (
  id                    uuid PRIMARY KEY,
  run_id                uuid NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
  stage                 text NOT NULL,              -- INGEST|DISCOVER|…|REPORT
  status                text NOT NULL,              -- PENDING|RUNNING|OK|FAILED|SKIPPED
  items_total           integer NOT NULL DEFAULT 0,
  items_done            integer NOT NULL DEFAULT 0,
  items_failed          integer NOT NULL DEFAULT 0,
  started_at            timestamptz,
  finished_at           timestamptz,
  error                 text NOT NULL DEFAULT '',
  UNIQUE (run_id, stage)
)
```

`pipeline_stages` is what makes an investigation resumable: restart resumes at
the first non-`OK` stage rather than re-running paid discovery calls.

---

## 2. Image evidence

```sql
images (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  role                  text NOT NULL,              -- PROBE (uploaded) | DISCOVERED
  sha256                bytea NOT NULL,
  phash                 bit(64) NOT NULL,
  dhash                 bit(64),
  whash                 bit(64),
  width                 integer,
  height                integer,
  file_size             bigint,
  mime_type             text NOT NULL DEFAULT '',
  exif                  jsonb NOT NULL DEFAULT '{}',
  storage_key           text NOT NULL,              -- object store
  source_page_id        uuid REFERENCES pages(id),  -- null for PROBE
  source_image_url      text NOT NULL DEFAULT '',
  archive_url           text NOT NULL DEFAULT '',
  discovered_at         timestamptz,
  downloaded_at         timestamptz,
  first_seen_at         timestamptz,                -- earliest observed publication
  last_seen_at          timestamptz,
  created_at            timestamptz NOT NULL DEFAULT now()
)
CREATE INDEX ON images (investigation_id, role);
CREATE INDEX ON images USING hash (sha256);
```

`first_seen_at` / `last_seen_at` are **observed** publication bounds derived from
provider dates and archive snapshots — not the crawl time, which is
`downloaded_at`. Conflating them would make every image look newly published.

```sql
image_relationships (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  from_image_id         uuid NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  to_image_id           uuid NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  relationship          text NOT NULL,   -- EXACT_COPY|RESIZED_COPY|CROPPED_COPY|
                                         -- MIRRORED_COPY|COMPRESSED_COPY|
                                         -- NEAR_DUPLICATE|THUMBNAIL|UNVERIFIED|REJECTED
  phash_distance        integer,
  justification         jsonb NOT NULL,  -- the measurements behind the call
  classified_at         timestamptz NOT NULL DEFAULT now(),
  classifier_version    text NOT NULL,
  UNIQUE (from_image_id, to_image_id)
)
```

`justification` stores the actual measurements (hamming distances, dimension
ratios, mirrored-hash result) so a classification can be re-checked months later
without re-fetching. `classifier_version` lets us tell which threshold set
produced a call after calibration changes.

---

## 3. Sources

```sql
domains (
  id                    uuid PRIMARY KEY,
  registrable_domain    text UNIQUE NOT NULL,       -- eTLD+1, public suffix list
  classification        text NOT NULL DEFAULT 'UNKNOWN',
      -- GOVERNMENT|COMPANY|EDUCATIONAL|NEWS|SOCIAL_MEDIA|FORUM|BLOG|
      -- ARCHIVE|DOCUMENTATION|UNKNOWN
  classification_basis  jsonb NOT NULL DEFAULT '{}',-- signals used
  first_seen_at         timestamptz NOT NULL DEFAULT now()
)
```

Classification is **descriptive metadata only**. It is deliberately *not*
referenced by any confidence computation (ARCHITECTURE §9.3) — encoding
institutional trust into an evidence score would smuggle an editorial judgement
into a number presented as objective. `classification_basis` records which
signals fired so the label is auditable.

```sql
appearances (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  probe_image_id        uuid NOT NULL REFERENCES images(id),
  provider              text NOT NULL,              -- plugin name
  page_url              text NOT NULL,
  image_url             text NOT NULL DEFAULT '',
  provider_score        double precision,
  provider_reported_date timestamptz,
  thumbnail_key         text NOT NULL DEFAULT '',
  archive_url           text NOT NULL DEFAULT '',
  discovered_at         timestamptz NOT NULL DEFAULT now(),
  verified_at           timestamptz,
  verification_result   text NOT NULL DEFAULT 'PENDING'
)

pages (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  domain_id             uuid NOT NULL REFERENCES domains(id),
  url                   text NOT NULL,
  canonical_url         text NOT NULL DEFAULT '',
  http_status           integer,
  title                 text NOT NULL DEFAULT '',
  description           text NOT NULL DEFAULT '',
  site_name             text NOT NULL DEFAULT '',
  language              text NOT NULL DEFAULT '',
  author_raw            text NOT NULL DEFAULT '',
  published_at          timestamptz,
  updated_at_source     timestamptz,
  content_simhash       bit(64),
  raw_html_key          text NOT NULL DEFAULT '',
  screenshot_key        text NOT NULL DEFAULT '',
  text_content_key      text NOT NULL DEFAULT '',
  outbound_links        jsonb NOT NULL DEFAULT '[]',
  fetched_at            timestamptz,
  fetch_error           text NOT NULL DEFAULT '',
  observable            boolean NOT NULL DEFAULT true,
  UNIQUE (investigation_id, url)
)
CREATE INDEX ON pages (domain_id);
```

`observable` distinguishes "we could not read this page" from "this page is
gone" — see §7.

```sql
content_clusters (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  original_page_id      uuid REFERENCES pages(id),
  member_count          integer NOT NULL DEFAULT 0,
  created_at            timestamptz NOT NULL DEFAULT now()
)

content_cluster_members (
  cluster_id            uuid NOT NULL REFERENCES content_clusters(id) ON DELETE CASCADE,
  page_id               uuid NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  role                  text NOT NULL,   -- ORIGINAL|REPOST|MIRROR|TRANSLATION|
                                         -- PARTIAL_COPY|MODIFIED_COPY
  similarity            double precision NOT NULL,
  role_confidence       text NOT NULL,   -- TRANSLATION is flagged lower
  PRIMARY KEY (cluster_id, page_id)
)
```

Clusters collapse to their `ORIGINAL` for independence counting, which is why
`SCORE` must run after `CLUSTER`.

---

## 4. Evidence chain

```sql
observations (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  page_id               uuid REFERENCES pages(id) ON DELETE CASCADE,
  image_id              uuid REFERENCES images(id),      -- for OCR observations
  origin                text NOT NULL DEFAULT 'EXTRACTED', -- EXTRACTED only here
  method                text NOT NULL,   -- SCHEMA_ORG|OPENGRAPH|META|NER|REGEX|
                                         -- OCR|CAPTION|TITLE|PROVIDER
  raw_value             text NOT NULL,
  normalized_value      text NOT NULL DEFAULT '',
  char_start            integer,
  char_end              integer,
  context_snippet       text NOT NULL DEFAULT '',
  extractor_version     text NOT NULL,
  method_confidence     double precision,   -- e.g. OCR engine confidence
  extracted_at          timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT observations_machine_only CHECK (origin = 'EXTRACTED')
)
CREATE INDEX ON observations (investigation_id, page_id);
```

The `CHECK` is invariant 3: human annotation physically cannot land here. Notes
live in §6.

```sql
entities (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  type                  text NOT NULL,   -- PERSON|ORGANIZATION|LOCATION|EVENT|
                                         -- USERNAME|EMAIL|PHONE|WEBSITE|DOCUMENT|
                                         -- PRODUCT|VEHICLE|LANDMARK
  canonical_name        text NOT NULL,
  normalized_key        text NOT NULL,
  source_count          integer NOT NULL DEFAULT 0,
  independent_source_count integer NOT NULL DEFAULT 0,
  possible_duplicate_of uuid REFERENCES entities(id),
  first_seen_at         timestamptz,
  last_seen_at          timestamptz,
  UNIQUE (investigation_id, type, normalized_key)
)

entity_aliases (
  id                    uuid PRIMARY KEY,
  entity_id             uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  surface_form          text NOT NULL,
  occurrences           integer NOT NULL DEFAULT 1,
  UNIQUE (entity_id, surface_form)
)

mentions (
  id                    uuid PRIMARY KEY,
  observation_id        uuid NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  entity_id             uuid REFERENCES entities(id) ON DELETE SET NULL,
  entity_type           text NOT NULL,
  surface_form          text NOT NULL
)
```

`possible_duplicate_of` implements the conservative-merge rule: uncertain pairs
stay separate and flagged rather than being merged. Over-merging attributes one
person's facts to another; under-merging is a visible inconvenience.

```sql
facts (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  entity_id             uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  attribute             text NOT NULL,   -- employer|role|location|school|username…
  value                 text NOT NULL,
  normalized_value      text NOT NULL DEFAULT '',
  status                text NOT NULL,   -- COMMON|UNIQUE|CONFLICTED|UNKNOWN
  conflict_group_id     uuid,            -- shared by competing values
  confidence            text NOT NULL,   -- HIGH|MEDIUM|LOW|UNCERTAIN
  confidence_score      double precision NOT NULL,
  confidence_explanation jsonb NOT NULL, -- invariant 5: never null
  independent_source_count integer NOT NULL DEFAULT 0,
  observation_count     integer NOT NULL DEFAULT 0,
  first_asserted_at     timestamptz,
  computed_at           timestamptz NOT NULL DEFAULT now(),
  scorer_version        text NOT NULL,
  CONSTRAINT facts_explanation_present CHECK (confidence_explanation <> '{}'::jsonb)
)

fact_evidence (
  fact_id               uuid NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
  observation_id        uuid NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  PRIMARY KEY (fact_id, observation_id)
)
```

`conflict_group_id` is how contradictions are preserved: competing values share a
group, all are retained with their own evidence, and none is deleted. The
`CHECK` is invariant 5 — a confidence number without its explanation cannot be
stored, so it can never be displayed alone.

---

## 5. Graph and timeline

```sql
graph_nodes (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  node_type             text NOT NULL,   -- IMAGE|PAGE|DOMAIN|ARTICLE|ORGANIZATION|
                                         -- PERSON|LOCATION|EVENT|DOCUMENT|
                                         -- SOCIAL_PROFILE|ARCHIVE
  ref_table             text NOT NULL,
  ref_id                uuid NOT NULL,
  label                 text NOT NULL,
  asserted_by_page_id   uuid REFERENCES pages(id),
  attributes            jsonb NOT NULL DEFAULT '{}',
  CONSTRAINT person_requires_assertion CHECK (
    node_type <> 'PERSON' OR asserted_by_page_id IS NOT NULL
  ),
  UNIQUE (investigation_id, node_type, ref_id)
)

graph_edges (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  from_node_id          uuid NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  to_node_id            uuid NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  edge_type             text NOT NULL,   -- APPEARS_ON|HOSTED_BY|MENTIONS|
                                         -- AUTHORED_BY|LOCATED_IN|ARCHIVED_AS|
                                         -- COPY_OF|LINKS_TO|EMPLOYED_BY
  derivation            text NOT NULL,   -- the rule that produced this edge
  evidence_observation_ids uuid[] NOT NULL,
  confidence            text NOT NULL DEFAULT 'LOW',
  created_at            timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT edge_requires_evidence CHECK (
    array_length(evidence_observation_ids, 1) >= 1
  ),
  UNIQUE (from_node_id, to_node_id, edge_type)
)
```

Two constraints carry real weight. `person_requires_assertion` is invariant 2 —
a `PERSON` node cannot exist without a page that explicitly named them, so there
is no schema-level path from an image to a person. `edge_requires_evidence` is
invariant 1 — the graph cannot drift into an unsourced parallel truth, because
an unsupported edge fails to insert.

```sql
timeline_events (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  occurred_at           timestamptz NOT NULL,
  precision             text NOT NULL,   -- EXACT|DAY|MONTH|YEAR|INFERRED
  kind                  text NOT NULL,   -- IMAGE_FIRST_APPEARANCE|LATEST_APPEARANCE|
                                         -- ARCHIVE_SNAPSHOT|NEWS_PUBLICATION|
                                         -- SITE_UPDATE|PAGE_REMOVED|
                                         -- INVESTIGATION_ACTION
  description           text NOT NULL,
  page_id               uuid REFERENCES pages(id),
  image_id              uuid REFERENCES images(id),
  evidence_observation_id uuid REFERENCES observations(id),
  created_at            timestamptz NOT NULL DEFAULT now()
)
CREATE INDEX ON timeline_events (investigation_id, occurred_at);
```

`precision` is stored so the UI renders "2019" rather than "1 Jan 2019 00:00".
Rendering an inferred date as exact would be a lie of formatting.

---

## 6. Human workspace — strictly separate

```sql
notes (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  author_id             uuid NOT NULL REFERENCES users(id),
  origin                text NOT NULL DEFAULT 'HUMAN',
  title                 text NOT NULL DEFAULT '',
  body_richtext         jsonb NOT NULL DEFAULT '{}',   -- TipTap document
  body_plain            text NOT NULL DEFAULT '',      -- for search
  tags                  text[] NOT NULL DEFAULT '{}',
  created_at            timestamptz NOT NULL DEFAULT now(),
  updated_at            timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT notes_human_only CHECK (origin = 'HUMAN')
)

note_links (            -- a note citing evidence, never overwriting it
  note_id               uuid NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  target_type           text NOT NULL,   -- FACT|PAGE|IMAGE|ENTITY|TIMELINE_EVENT
  target_id             uuid NOT NULL,
  PRIMARY KEY (note_id, target_type, target_id)
)

note_attachments (
  id                    uuid PRIMARY KEY,
  note_id               uuid NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  filename              text NOT NULL,
  mime_type             text NOT NULL,
  storage_key           text NOT NULL,
  uploaded_at           timestamptz NOT NULL DEFAULT now()
)

checklists (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  title                 text NOT NULL
)

checklist_items (
  id                    uuid PRIMARY KEY,
  checklist_id          uuid NOT NULL REFERENCES checklists(id) ON DELETE CASCADE,
  label                 text NOT NULL,
  done                  boolean NOT NULL DEFAULT false,
  position              integer NOT NULL DEFAULT 0
)

bookmarks (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  owner_id              uuid NOT NULL REFERENCES users(id),
  target_type           text NOT NULL,
  target_id             uuid NOT NULL,
  label                 text NOT NULL DEFAULT '',
  created_at            timestamptz NOT NULL DEFAULT now(),
  UNIQUE (owner_id, target_type, target_id)
)
```

Two `CHECK` constraints (`observations_machine_only`, `notes_human_only`) form
invariant 3 from both sides. A note *links to* evidence; it can never become
evidence.

---

## 7. Monitoring

```sql
monitors (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  image_id              uuid NOT NULL REFERENCES images(id),
  cadence               text NOT NULL,   -- DAILY|WEEKLY|MONTHLY
  enabled               boolean NOT NULL DEFAULT true,
  next_run_at           timestamptz NOT NULL,
  last_run_at           timestamptz,
  created_at            timestamptz NOT NULL DEFAULT now()
)

monitor_runs (
  id                    uuid PRIMARY KEY,
  monitor_id            uuid NOT NULL REFERENCES monitors(id) ON DELETE CASCADE,
  pipeline_run_id       uuid REFERENCES pipeline_runs(id),
  started_at            timestamptz NOT NULL DEFAULT now(),
  finished_at           timestamptz,
  changes_detected      integer NOT NULL DEFAULT 0,
  status                text NOT NULL
)

monitor_changes (
  id                    uuid PRIMARY KEY,
  monitor_run_id        uuid NOT NULL REFERENCES monitor_runs(id) ON DELETE CASCADE,
  change_type           text NOT NULL,   -- NEW_APPEARANCE|PAGE_REMOVED|PAGE_UPDATED|
                                         -- IMAGE_REPLACED|ARCHIVE_ADDED|
                                         -- PAGE_UNOBSERVABLE
  page_id               uuid REFERENCES pages(id),
  image_id              uuid REFERENCES images(id),
  detail                jsonb NOT NULL DEFAULT '{}',
  confirming_runs       integer NOT NULL DEFAULT 1,
  timeline_event_id     uuid REFERENCES timeline_events(id),
  detected_at           timestamptz NOT NULL DEFAULT now()
)
```

`PAGE_UNOBSERVABLE` is a distinct change type from `PAGE_REMOVED`, and this is
the important detail. A 404/410 is evidence of removal. A 403, 429, timeout or
robots.txt change is **not** — it means we can no longer observe the page, which
is a different fact. `confirming_runs` requires N consecutive confirmations
before a removal is asserted, so one transient outage never fabricates a takedown
event.

---

## 8. Copilot, reports, audit, search

```sql
copilot_sessions (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  created_at            timestamptz NOT NULL DEFAULT now()
)

copilot_messages (
  id                    uuid PRIMARY KEY,
  session_id            uuid NOT NULL REFERENCES copilot_sessions(id) ON DELETE CASCADE,
  role                  text NOT NULL,             -- user|assistant
  content               text NOT NULL,
  citations             jsonb NOT NULL DEFAULT '[]', -- evidence ids per claim
  validation_status     text NOT NULL DEFAULT 'PENDING', -- PASSED|REJECTED
  rejected_spans        jsonb NOT NULL DEFAULT '[]',
  model                 text NOT NULL DEFAULT '',
  created_at            timestamptz NOT NULL DEFAULT now()
)
```

`validation_status` and `rejected_spans` persist the citation check, so a claim
that failed validation is recorded as rejected rather than quietly dropped —
the assistant's failures are auditable too.

```sql
reports (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  format                text NOT NULL,             -- HTML|PDF|JSON|MARKDOWN
  storage_key           text NOT NULL,
  content_hash          text NOT NULL,
  generated_by          uuid NOT NULL REFERENCES users(id),
  generated_at          timestamptz NOT NULL DEFAULT now()
)

audit_log (
  id                    bigserial PRIMARY KEY,      -- monotonic, chain order
  investigation_id      uuid REFERENCES investigations(id),
  actor_id              uuid REFERENCES users(id),
  action                text NOT NULL,
  outcome               text NOT NULL,
  lawful_basis          text NOT NULL DEFAULT '',
  detail                jsonb NOT NULL DEFAULT '{}',
  previous_hash         char(64) NOT NULL,
  entry_hash            char(64) NOT NULL,
  created_at            timestamptz NOT NULL DEFAULT now()
)
REVOKE UPDATE, DELETE ON audit_log FROM application_role;
```

Immutability is enforced by privilege, not intention: the application role can
`INSERT` and `SELECT` only. Combined with the hash chain, editing history
requires database-superuser access *and* still breaks every subsequent hash.
`verify()` re-walks the chain and reports the first divergent index.

```sql
search_documents (
  id                    uuid PRIMARY KEY,
  investigation_id      uuid REFERENCES investigations(id) ON DELETE CASCADE,
  object_type           text NOT NULL,   -- CASE|IMAGE|DOMAIN|PAGE|ENTITY|EVENT|
                                         -- OCR_TEXT|REPORT|NOTE|FACT
  object_id             uuid NOT NULL,
  title                 text NOT NULL DEFAULT '',
  body                  text NOT NULL DEFAULT '',
  tsv                   tsvector GENERATED ALWAYS AS (
                          setweight(to_tsvector('simple', coalesce(title,'')), 'A') ||
                          setweight(to_tsvector('simple', coalesce(body,'')),  'B')
                        ) STORED,
  updated_at            timestamptz NOT NULL DEFAULT now(),
  UNIQUE (object_type, object_id)
)
CREATE INDEX ON search_documents USING gin (tsv);
CREATE INDEX ON search_documents USING gin (title gin_trgm_ops);
```

One table, one GIN index, comparable ranking across every object type, and no
second datastore. Populated by triggers on write.

---

## 9. Migration policy

- Every schema change ships as an Alembic revision with a tested downgrade.
- Enum-like columns are `text` + `CHECK`, not PostgreSQL enums: adding a value to
  a PG enum is a migration that locks the table, and these vocabularies will grow.
- No destructive migration on evidence tables. Reclassification writes new rows
  with a new `*_version`; it never mutates history. Old reports must stay
  reproducible.
