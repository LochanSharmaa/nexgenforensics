import { useEffect, useState } from "react";
import {
  IIE_BASE,
  discover,
  getFindings,
  listProviders,
  provenanceEnabled,
  traceImage,
} from "../../services/provenanceApi";

/**
 * Public-web provenance for the probe already loaded on this page.
 *
 * Organised around the questions an investigator asks — *where has this been
 * published, what do those pages say, when did it first appear, do the sources
 * disagree* — not around the pipeline that answers them. Stage names and hash
 * digests are plumbing; they sit in a collapsed technical section, and only
 * because a finding has to be reproducible, not because anyone reads them.
 *
 * The complement to face search, never a substitute. Face search asks who this
 * face resembles among lawfully enrolled subjects. This asks where this *file*
 * has been published. No facial analysis happens here at any point.
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

  useEffect(() => {
    if (!provenanceEnabled) return;
    listProviders().then(setProviders).catch(() => setProviders([]));
  }, []);

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

  async function handleTrace() {
    setBusy(true);
    setError("");
    setFindings(null);

    try {
      const started = await traceImage({
        file,
        caseRef: `PROV-${caseReference || "adhoc"}-${Date.now().toString(36)}`,
        title: `Provenance — ${file?.name || "probe image"}`,
        lawfulBasis,
        purpose,
        onStep: setStep,
      });
      setTechnical(started);

      setStep("Asking discovery providers…");
      const result = await discover(started.investigation.id, {
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

      {file && !findings && (
        <>
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
