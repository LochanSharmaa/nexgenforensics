# Security & governance

[← Back to README](../README.md)

## Biometric data handling

**Templates are encrypted at rest** with AES-256-GCM under a per-tenant HKDF
subkey, with the tenant id bound in as additional authenticated data. A
ciphertext moved between tenant rows fails to decrypt rather than silently
matching under the wrong tenant.

HKDF is used per tenant rather than PBKDF2 per record. The earlier design ran
600k PBKDF2 iterations on every encrypt *and* every decrypt, which would add
roughly a second of CPU to each template touched and make a gallery load of any
size unusable. PBKDF2 remains on the `from_passphrase` path, where the input is
genuinely low-entropy.

**Templates never leave the server.** An ArcFace embedding can be inverted into a
recognisable approximation of the face it came from, so it is treated as
equivalent to the photograph. The API returns template *metadata* only.

**Erasure is real.** Deleting a subject removes templates, enrolment images and
index entries outright — not a soft-delete flag. Search history is retained,
because deleting it would destroy the audit trail; candidate rows keep the
subject id only.

Biometrics are irrevocable. A leaked template cannot be reissued like a
password, which is why none of this is optional.

## Tenant isolation

Isolation is **structural, not advisory**. Each tenant's vectors live in a
separate matrix and `search()` takes the tenant id as a required argument, so
there is no code path that can compare a probe against another tenant's
templates.

A filter applied after a global search would be one forgotten predicate away
from a cross-tenant biometric leak. The tenant always comes from the verified
credential, never from a request body or header.

Cross-tenant lookups return `404`, not `403`: a distinguishable `403` would
confirm that a given case or subject id exists.

## Authentication

- **Argon2id** password hashing (OWASP interactive parameters), with transparent
  rehash when cost parameters change
- **Constant-time login** — a verification always runs, even with no matching
  user, so response timing cannot enumerate registered accounts
- Identical error text for unknown user and wrong password
- **Short-lived access tokens** with separate refresh tokens; a refresh token is
  rejected where an access token is expected, so a long-lived credential cannot
  silently become a session
- **API keys** hashed with SHA-256 (32 bytes of CSPRNG output has no low-entropy
  secret to slow-hash), shown once, never recoverable. Revocation deactivates
  rather than deletes, so audit references stay resolvable

Roles: `investigator` → `supervisor` → `admin`. Enrolment requires supervisor,
because it determines who the system is *capable* of finding.

## Audit trail

Every consequential action is recorded in a hash chain, per tenant, mirrored to
JSONL on disk. Each record's hash covers its content, its explicit chain
position, and the previous record's hash.

Chain position is an explicit `sequence` column, **not** a timestamp ordering.
`created_at` has millisecond resolution at best, so a burst of writes shares a
tick; ordering by time with a random-UUID tiebreaker made verification
non-deterministic and could report tampering on an untouched log. For a forensic
system that false positive is close to the worst possible defect, since it
renders real detections indistinguishable from noise.

**What this proves:** no record has been altered or removed since it was
written. Editing or deleting one breaks verification from that point onward, and
a gap in the sequence is reported as a removed record.

**What it does not prove:** completeness. Someone with database access could
truncate the tail. Ship the mirrored JSONL to write-once storage if you need
that guarantee.

Any authenticated user can read their tenant's audit trail, including entries
about themselves. A log only administrators can see is far easier to quietly
misuse.

## Deliberate refusals

**Server-side URL import returns 501.** Fetching a caller-supplied URL from
inside the service is a server-side request forgery primitive that would reach
internal addresses and cloud metadata endpoints using the server's network
position. Enabling it needs a host allow-list, DNS-rebinding protection and
egress limits.

**Only a human can confirm a candidate.** The engine has no code path that
writes `confirmed`. Automated face recognition produces investigative leads.

**Every search requires a stated lawful basis**, recorded verbatim in the audit
chain. The system cannot judge whether a search is lawful; it can ensure someone
had to state a reason and that the statement is preserved.

## Transport & headers

CORS origins are explicit — never `*`, which is both invalid and dangerous with
credentialed requests. Every response carries `Cache-Control: no-store`
(biometric findings must not sit in a shared cache), `X-Content-Type-Options`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, a correlation
`X-Request-ID`, and HSTS in production.

Unhandled exceptions return a generic message plus the request id. Stack traces
routinely leak schema, file paths and occasionally credentials.

## Not claimed

The liveness, deepfake and morphing screens are **heuristics, not certified
detection**. None has been evaluated against ISO/IEC 30107-3, and a determined
attacker defeats all three. They look for cheap artefacts — texture collapse,
screen moiré, spectral checkerboarding. Use them to prioritise examiner
attention, never to conclude that media is authentic.

---

## Before production

- [ ] Set and **back up** `NEXGEN_JWT_SECRET` and `NEXGEN_TEMPLATE_KEY`
- [ ] Calibrate thresholds on representative imagery
- [ ] Run `tests/test_recognition_engine.py` against real faces
- [ ] Move to PostgreSQL; SQLite will not survive concurrent load
- [ ] Terminate TLS at the ingress and restrict `NEXGEN_CORS_ORIGINS`
- [ ] Complete legal, privacy and DPIA review for your jurisdiction
- [ ] **Evaluate demographic performance differentials on your own population**
- [ ] Define retention and deletion policy; set `NEXGEN_PROBE_RETENTION_DAYS`
- [ ] Ship audit JSONL to append-only storage
- [ ] Establish examiner training and an adjudication standard

The demographic item is not a formality. Face recognition error rates vary
across demographic groups. A system deployed without measuring that on its own
population will distribute its errors unevenly across the people it is used on,
and nobody will notice.
