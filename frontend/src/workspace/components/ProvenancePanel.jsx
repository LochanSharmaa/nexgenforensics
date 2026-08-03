import { useCallback, useEffect, useRef, useState } from "react";
import {
  IIE_BASE,
  analyzeImage,
  discover,
  getFindings,
  listProviders,
  openCase,
  provenanceEnabled,
  startRun,
} from "../../services/provenanceApi";

/**
 * Public-web provenance for the probe already loaded on this page.
 *
 * Organised around the questions an investigator asks — *what does this picture
 * show, where has it been published, what do those pages say, when did it first
 * appear, do the sources disagree* — not around the pipeline that answers them.
 * Stage names and hash digests are plumbing; they sit in a collapsed technical
 * section, and only because a finding has to be reproducible, not because
 * anyone reads them.
 *
 * The two questions are kept visibly apart because they carry different weight.
 * *What the image shows* is an observation — checkable by looking at the
 * picture, and true of the image whether or not anything else corroborates it.
 * *Where it appears* is a claim about the world, and needs sources. Presenting
 * them as one list would let a transcribed sign read as an established fact.
 *
 * The complement to face search, never a substitute. Face search asks who this
 * face resembles among lawfully enrolled subjects. This asks what this *file*
 * contains and where it has been published. No facial analysis happens here at
 * any point, and the image-reading stage will not name a person even when it
 * can plainly see one.
 */

const CONFIDENCE_TONE = { high: "good", medium: "review", low: "neutral" };

export function ProvenancePanel({ file, lawfulBasis, purpose, caseReference }) {
  const [providers, setProviders] = useState([]);
  const [urls, setUrls] = useState("");
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState("");
  const [error, setError] = useState("");
  const [findings, setFindings] = useState(null);
  const [technical, setTechnical] = useState(null);
  const [showTechnical, setShowTechnical] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [analysing, setAnalysing] = useState(false);
  const [analysisError, setAnalysisError] = useState("");

  // The opened case, held in a ref as well as in state. Two buttons can each
  // need it, and reading it from state inside a handler that opened it moments
  // earlier would see the pre-update value and open a second, duplicate case.
  const caseRef = useRef(null);

  useEffect(() => {
    if (!provenanceEnabled) return;
    listProviders().then(setProviders).catch(() => setProviders([]));
  }, []);

  // A different image is a different case. Without this, analysis from the
  // previous probe would sit under the new one — evidence attributed to the
  // wrong picture, which is the worst failure this panel could have.
  useEffect(() => {
    caseRef.current = null;
    setAnalysis(null);
    setFindings(null);
    setTechnical(null);
    setError("");
    setAnalysisError("");
  }, [file]);

  const ensureCase = useCallback(
    async (onStep) => {
      if (caseRef.current) return caseRef.current;
      const opened = await openCase({
        file,
        caseRef: `PROV-${caseReference || "adhoc"}-${Date.now().toString(36)}`,
        title: `Provenance — ${file?.name || "probe image"}`,
        lawfulBasis,
        purpose,
        onStep,
      });
      caseRef.current = opened;
      setTechnical(opened);
      return opened;
    },
    [file, caseReference, lawfulBasis, purpose],
  );

  if (!provenanceEnabled) {
    return (
      <section className="wk-card">
        <h2>Public-web provenance</h2>
        <p>
          Not configured. Set <span className="wk-mono">VITE_IIE_API_BASE</span> to the
          Image Intelligence Engine origin to enable it.
        </p>
      </section>
    );
  }

  const configured = providers.filter((p) => p.configured);
  const missing = providers.filter((p) => !p.configured);

  async function handleAnalyse() {
    setAnalysing(true);
    setAnalysisError("");

    try {
      const opened = await ensureCase(setStep);
      setStep("");
      const result = await analyzeImage(opened.investigation.id, {
        imageId: opened.image.id,
      });
      setAnalysis(result);
    } catch (visionError) {
      setAnalysisError(visionError.message);
    } finally {
      setAnalysing(false);
      setStep("");
    }
  }

  async function handleTrace() {
    setBusy(true);
    setError("");
    setFindings(null);

    try {
      const opened = await ensureCase(setStep);

      setStep("Starting discovery run…");
      await startRun(opened.investigation.id);

      setStep("Asking discovery providers…");
      const result = await discover(opened.investigation.id, {
        urls: urls
          .split(/[\s,]+/)
          .map((u) => u.trim())
          .filter(Boolean),
      });
      setFindings(result);
    } catch (traceError) {
      setError(traceError.message);
    } finally {
      setBusy(false);
      setStep("");
    }
  }

  return (
    <section className="wk-card">
      <h2>Public-web provenance</h2>
      <p>
        Where has this <strong>photograph</strong> been published, and what do those
        pages say? Reverse image search over file hashes — no facial recognition, no
        biometric data.
      </p>

      {!file && <div className="wk-empty">Select a probe image above to trace it.</div>}

      {file && (
        <VisionSection
          analysis={analysis}
          analysing={analysing}
          error={analysisError}
          step={step}
          disabled={busy || !lawfulBasis?.trim()}
          onAnalyse={handleAnalyse}
        />
      )}

      {file && !findings && (
        <>
          <h3 style={{ margin: "26px 0 8px", fontSize: 15 }}>Where this image appears</h3>
          <label className="wk-field">
            <span>Known URLs (optional)</span>
            <textarea
              rows={2}
              value={urls}
              onChange={(event) => setUrls(event.target.value)}
              placeholder="https://example.com/article&#10;https://example.com/staff"
            />
            <small>
              Pages you already suspect carry this image. Checked directly — this is
              targeted corroboration, not a crawl.
            </small>
          </label>

          <button
            type="button"
            className="wk-button"
            disabled={busy || !lawfulBasis?.trim()}
            onClick={handleTrace}
          >
            {busy ? step || "Working…" : "Trace this image"}
          </button>

          {!lawfulBasis?.trim() && (
            <small style={{ display: "block", marginTop: 8, color: "var(--muted)" }}>
              State a lawful basis above first — it is recorded in the provenance
              service&rsquo;s own audit chain too.
            </small>
          )}

          <CoverageNote configured={configured} missing={missing} />
        </>
      )}

      {error && <div className="wk-error" style={{ marginTop: 12 }}>{error}</div>}

      {findings && (
        <>
          <FindingsSummary summary={findings.summary} searched={findings.searched} />

          <h3 style={{ margin: "22px 0 8px", fontSize: 15 }}>Where this image appears</h3>
          {findings.findings.length === 0 ? (
            <div className="wk-empty">
              No public appearances found. With only the providers currently
              configured, that means this image was not among the pages checked —
              not that it appears nowhere online.
            </div>
          ) : (
            <div className="wk-table-wrap">
              <table className="wk-table">
                <thead>
                  <tr>
                    <th>Source</th>
                    <th>Page</th>
                    <th>Match</th>
                    <th>Found by</th>
                    <th>Verified</th>
                  </tr>
                </thead>
                <tbody>
                  {findings.findings.map((finding) => (
                    <tr key={`${finding.url}-${finding.match_kind}`}>
                      <td className="wk-mono">{finding.site}</td>
                      <td>
                        <a href={finding.url} target="_blank" rel="noreferrer noopener">
                          {finding.title || finding.url}
                        </a>
                        {(finding.reported_date || finding.archive_url) && (
                          <small style={{ display: "block", color: "var(--muted)" }}>
                            {finding.reported_date && (
                              <>first archived {finding.reported_date}</>
                            )}
                            {finding.archive_url && (
                              <>
                                {finding.reported_date ? " · " : ""}
                                <a
                                  href={finding.archive_url}
                                  target="_blank"
                                  rel="noreferrer noopener"
                                  title="A durable snapshot. Live pages get edited or removed mid-investigation; this one will still resolve."
                                >
                                  archived copy
                                </a>
                              </>
                            )}
                          </small>
                        )}
                      </td>
                      <td>
                        <span className={`wk-chip ${CONFIDENCE_TONE[finding.confidence]}`}>
                          {finding.match_label}
                        </span>
                      </td>
                      <td className="wk-mono">{finding.provider}</td>
                      <td>
                        <span className="wk-chip neutral">{finding.verification}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <PlaceholderSection
            title="What these pages say"
            body="Names, organisations, roles and locations stated on the pages above — each shown with the page that states it and how many independent sources agree. A name here is always a claim by that page, never an identification by this system."
          />
          {findings.summary.earliest_appearance ? (
            <>
              <h3 style={{ margin: "22px 0 8px", fontSize: 15 }}>Timeline</h3>
              <div className="wk-metrics">
                <div className="wk-metric">
                  <span>Earliest archived</span>
                  <strong>{findings.summary.earliest_appearance}</strong>
                  <small>independent capture, not a claimed date</small>
                </div>
                <div className="wk-metric">
                  <span>Latest archived</span>
                  <strong>{findings.summary.latest_appearance}</strong>
                </div>
              </div>
            </>
          ) : (
            <PlaceholderSection
              title="Timeline"
              body="Earliest and latest capture, archive snapshots, and any republication in between. Dates come from third-party captures rather than a page's own claimed date, because a page owner can edit the latter."
            />
          )}
          <PlaceholderSection
            title="Conflicting sources"
            body="Where sources disagree, every version is kept with its own evidence. Nothing is silently resolved in favour of one."
          />

          <CoverageNote configured={configured} missing={missing} />

          {technical && (
            <>
              <button
                type="button"
                className="wk-button ghost small"
                style={{ marginTop: 16 }}
                onClick={() => setShowTechnical((v) => !v)}
              >
                {showTechnical ? "Hide" : "Show"} technical detail
              </button>
              {showTechnical && (
                <div className="wk-metrics" style={{ marginTop: 12 }}>
                  <div className="wk-metric">
                    <span>Provenance case</span>
                    <strong className="wk-mono">{technical.investigation.case_id}</strong>
                  </div>
                  <div className="wk-metric">
                    <span>SHA-256</span>
                    <strong className="wk-mono">
                      {technical.image.sha256.slice(0, 16)}…
                    </strong>
                    <small>identifies the exact file</small>
                  </div>
                  <div className="wk-metric">
                    <span>Perceptual hash</span>
                    <strong className="wk-mono">{technical.image.phash}</strong>
                    <small>survives resize and re-encode</small>
                  </div>
                  <div className="wk-metric">
                    <span>Service</span>
                    <strong className="wk-mono">{IIE_BASE}</strong>
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}

/* ------------------------------------------------------- what the image shows */

/**
 * Categories in the order an investigator works through them, with the label
 * each is worth to a reader rather than the enum name.
 *
 * Ordered by how far a category typically gets you: a transcribed sign or a
 * document reference is a searchable lead, a landmark narrows a place, and
 * "objects" is background. Anything the model returns under a category not
 * listed here still renders — dropping an observation because the vocabulary
 * moved would be losing evidence to a presentation detail.
 */
const CATEGORY_LABELS = [
  ["TEXT", "Text in the image"],
  ["SIGN", "Signage"],
  ["DOCUMENT", "Documents"],
  ["LOGO", "Logos and wordmarks"],
  ["LANDMARK", "Landmarks"],
  ["LOCATION_CUE", "Location cues"],
  ["DATE", "Dates and times"],
  ["VEHICLE", "Vehicles"],
  ["OBJECT", "Objects"],
  ["VISUAL_CLUE", "Other visual detail"],
];

/** Only where the heading alone would be read as more than it is. */
const CATEGORY_NOTES = {
  LOCATION_CUE:
    "Cues that bear on place — script on signage, plate format, road markings. " +
    "Not a location. Where the photograph was taken is a claim, and needs sources.",
  LANDMARK: "A structure can be a replica. Treat as a lead, not a placement.",
  DOCUMENT: "A document can be forged. This records what is legible, not that it is genuine.",
};

/**
 * What the picture shows, kept separate from what the web says about it.
 *
 * Everything here is an observation: a statement about the image that a
 * reviewer can check by looking at the image. None of it is a fact about the
 * world, and the section says so plainly rather than relying on the reader to
 * remember the distinction.
 */
function VisionSection({ analysis, analysing, error, step, disabled, onAnalyse }) {
  const groups = analysis ? groupObservations(analysis.by_category) : [];
  const clues = analysis?.clues || [];
  const rejected = analysis?.rejected || [];

  return (
    <>
      <h3 style={{ margin: "22px 0 8px", fontSize: 15 }}>What the image shows</h3>

      {!analysis && !analysing && (
        <>
          <p style={{ margin: "0 0 12px", color: "var(--muted)", fontSize: 14 }}>
            Read the picture itself — signage, documents, logos, plate formats,
            anything legible. These become search leads. No person is identified:
            people are counted, never named.
          </p>
          <button
            type="button"
            className="wk-button ghost"
            disabled={disabled}
            onClick={onAnalyse}
          >
            Read this image
          </button>
          <small style={{ display: "block", marginTop: 8, color: "var(--muted)" }}>
            Around twenty seconds — it is a live model call.
          </small>
        </>
      )}

      {analysing && (
        <div className="wk-loading">
          {step || "Reading the image…"}
          <br />
          <small>This takes around twenty seconds.</small>
        </div>
      )}

      {error && <div className="wk-error" style={{ marginTop: 12 }}>{error}</div>}

      {analysis && !analysis.available && (
        <p className="wk-notice">
          Image reading is not configured. Set{" "}
          <span className="wk-mono">IIE_GEMINI_API_KEY</span> on the provenance
          service to enable it. Discovery below works without it.
        </p>
      )}

      {analysis?.available && analysis.error && (
        <div className="wk-error">The model call failed: {analysis.error}</div>
      )}

      {analysis?.available && !analysis.error && (
        <>
          <p className="wk-notice" style={{ marginBottom: 16 }}>
            <strong>These are observations about the image, not facts about the
            world.</strong>{" "}
            A sign can be a mock-up and a document can be forged. Each line below
            is checkable by looking at the picture; nothing here is corroborated
            until a source in the section beneath states it too.
          </p>

          <div className="wk-metrics" style={{ marginBottom: 18 }}>
            <div className="wk-metric">
              <span>Observations</span>
              <strong>{analysis.observation_count}</strong>
              <small>each sourced to the image</small>
            </div>
            <div className="wk-metric">
              <span>Search leads</span>
              <strong>{clues.length}</strong>
              <small>derived from what was read</small>
            </div>
            <div className="wk-metric">
              <span>People visible</span>
              <strong>{analysis.people_present}</strong>
              <small>a count only — never who</small>
            </div>
          </div>

          {groups.length === 0 ? (
            <div className="wk-empty">
              Nothing legible was found in this image. That is a result about the
              picture, not about the person in it.
            </div>
          ) : (
            groups.map(([key, label, items]) => (
              <ObservationGroup
                key={key}
                label={label}
                note={CATEGORY_NOTES[key]}
                items={items}
              />
            ))
          )}

          <ClueList clues={clues} />
          <RefusedList rejected={rejected} />

          <small style={{ display: "block", marginTop: 14, color: "var(--muted)" }}>
            Read by <span className="wk-mono">{analysis.model}</span> in{" "}
            {(analysis.duration_ms / 1000).toFixed(1)}s. Stored as observations
            against this image, so the reading can be reviewed against the
            picture later.
          </small>
        </>
      )}
    </>
  );
}

/** Known categories in reading order, then anything unrecognised, so a category
 *  added on the service side still reaches the screen. */
function groupObservations(byCategory = {}) {
  const known = new Set(CATEGORY_LABELS.map(([key]) => key));
  const ordered = CATEGORY_LABELS.filter(([key]) => byCategory[key]?.length).map(
    ([key, label]) => [key, label, byCategory[key]],
  );
  const extra = Object.entries(byCategory)
    .filter(([key, items]) => !known.has(key) && items?.length)
    .map(([key, items]) => [key, key.replace(/_/g, " ").toLowerCase(), items]);
  return [...ordered, ...extra];
}

function ObservationGroup({ label, note, items }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div className="wk-subhead" style={{ margin: "0 0 8px" }}>
        {label}
      </div>
      {note && (
        <small style={{ display: "block", marginBottom: 8, color: "var(--muted)" }}>
          {note}
        </small>
      )}
      <ul className="wk-observations">
        {items.map((item, index) => (
          <li key={`${item.value}-${index}`}>
            <span className="wk-observation-value">{item.value}</span>
            {!item.verbatim && (
              <span className="wk-chip review" title="Summarised rather than transcribed. Weaker evidence than a quoted reading.">
                paraphrased
              </span>
            )}
            {item.detail && <small>{item.detail}</small>}
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Queries the observations justify.
 *
 * Copy rather than a search link: running these sends text read off the
 * investigator's evidence to a third party, and that should be a deliberate act
 * on their part, not a click that looks like part of this page.
 */
function ClueList({ clues }) {
  if (!clues.length) return null;
  return (
    <>
      <div className="wk-subhead">Search leads</div>
      <small style={{ display: "block", marginBottom: 8, color: "var(--muted)" }}>
        Built from what was read, never invented. Paste into a search tool, or
        into the known-URLs box below once a lead resolves to a page.
      </small>
      <ul className="wk-clues">
        {clues.map((clue, index) => (
          <li key={`${clue.query}-${index}`}>
            <div>
              <code>{clue.query}</code>
              <small>{clue.rationale}</small>
            </div>
            <button
              type="button"
              className="wk-button ghost small"
              onClick={() => navigator.clipboard?.writeText(clue.query)}
              title="Copy this query"
            >
              Copy
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}

/**
 * What the guardrails refused.
 *
 * Shown, not hidden. A model that keeps reaching for an identification is a
 * fact about the model that an operator is entitled to see — and an empty
 * section here is itself worth reading, because it means the refusal path was
 * exercised and found nothing to refuse.
 */
function RefusedList({ rejected }) {
  if (!rejected.length) return null;
  return (
    <>
      <div className="wk-subhead">Refused by the guardrails</div>
      <small style={{ display: "block", marginBottom: 8, color: "var(--muted)" }}>
        The model produced these and the platform dropped them before they became
        evidence. Recorded in the audit trail too.
      </small>
      <ul className="wk-reason-list">
        {rejected.map((item, index) => (
          <li key={`${item.rule}-${index}`}>
            <code>{item.rule}</code>
            <span>
              {item.reason}
              {item.value && (
                <>
                  {" "}
                  <em style={{ color: "var(--muted)" }}>Dropped: “{item.value}”</em>
                </>
              )}
            </span>
          </li>
        ))}
      </ul>
    </>
  );
}

/* ------------------------------------------------------------------ findings */

function FindingsSummary({ summary, searched }) {
  return (
    <div className="wk-metrics" style={{ margin: "16px 0" }}>
      <div className="wk-metric">
        <span>Sources found</span>
        <strong>{summary.sources_found}</strong>
        <small>{searched ? "providers were asked" : "not searched yet"}</small>
      </div>
      <div className="wk-metric">
        <span>Distinct sites</span>
        <strong>{summary.distinct_sites}</strong>
        <small>one site is one source</small>
      </div>
      <div className="wk-metric">
        <span>Exact matches</span>
        <strong>{summary.exact_matches}</strong>
        <small>same image file</small>
      </div>
      <div className="wk-metric">
        <span>Similar only</span>
        <strong>{summary.similar_only}</strong>
        <small>look-alike, not the same image</small>
      </div>
    </div>
  );
}

/**
 * What was and was not asked. Shown before and after a search, because "found
 * nothing" and "never looked" are different facts and an investigator must not
 * have to infer which one they are reading.
 */
function CoverageNote({ configured, missing }) {
  if (!configured.length && !missing.length) return null;
  return (
    <p className="wk-notice" style={{ marginTop: 16 }}>
      {configured.length > 0 && (
        <>
          Searched with: <strong>{configured.map((p) => p.title).join(", ")}</strong>.{" "}
        </>
      )}
      {missing.length > 0 && (
        <>
          Not searched — no credentials for{" "}
          <strong>{missing.map((p) => p.title).join(", ")}</strong>. Set{" "}
          <span className="wk-mono">
            {missing.flatMap((p) => p.config_keys).join(", ")}
          </span>{" "}
          to widen coverage. Absence of a result here is not evidence of absence
          online.
        </>
      )}
    </p>
  );
}

function PlaceholderSection({ title, body }) {
  return (
    <>
      <h3 style={{ margin: "22px 0 8px", fontSize: 15 }}>{title}</h3>
      <div className="wk-empty">{body}</div>
    </>
  );
}
