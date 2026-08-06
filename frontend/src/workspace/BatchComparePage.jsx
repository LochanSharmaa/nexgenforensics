import { useEffect, useState } from "react";
import { fetchCaseReport, listCases, runBatch } from "../services/imatchApi";
import { ImageDropZone } from "./components/ImageDropZone";

const ACCEPTED = "image/jpeg,image/png,image/webp,image/bmp,image/tiff";

/**
 * One reference face compared against a set of uploaded images — the ad-hoc
 * "here is my suspect, check these stills" workflow. Nothing is enrolled and
 * the gallery is not involved; every uploaded image is compared 1:1 against
 * the reference by the batch endpoint (mode=one_to_many).
 */
export function BatchComparePage() {
  const [cases, setCases] = useState([]);
  const [referenceFile, setReferenceFile] = useState(null);
  const [probeFiles, setProbeFiles] = useState([]);
  const [caseId, setCaseId] = useState("");
  const [lawfulBasis, setLawfulBasis] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [generatingReport, setGeneratingReport] = useState("");

  useEffect(() => {
    listCases().then(setCases).catch(() => setCases([]));
  }, []);

  // Built server-side from every image recorded against the case, not from the
  // batch shown above: a report assembled from this page's state would describe
  // one run rather than the case.
  async function generateReport(format) {
    setError("");
    setGeneratingReport(format);
    try {
      const pdf = await fetchCaseReport(caseId, format);
      const caseRecord = cases.find((item) => item.id === caseId);
      const stem = format === "examination" ? "examination" : "case";
      const url = URL.createObjectURL(pdf);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${stem}-${caseRecord?.reference || caseId}.pdf`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (reportError) {
      setError(reportError.message);
    } finally {
      setGeneratingReport("");
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setRunning(true);
    setError("");
    setResult(null);
    try {
      setResult(
        await runBatch({
          mode: "one_to_many",
          referenceFile,
          probeFiles,
          lawfulBasis,
          caseId: caseId || null,
        }),
      );
    } catch (batchError) {
      setError(batchError.message);
    } finally {
      setRunning(false);
    }
  }

  const overLimit = probeFiles.length > 50;

  return (
    <>
      <div className="wk-page-head">
        <div>
          <h1>Reference vs set</h1>
          <p>
            Compare one reference face against a set of uploaded images. Nothing is enrolled
            and the gallery is not searched — each image is compared 1:1 against the reference.
          </p>
        </div>
      </div>

      {error && <div className="wk-error">{error}</div>}

      <section className="wk-card">
        <form onSubmit={handleSubmit}>
          <div className="wk-grid two">
            <ImageDropZone
              id="batch-reference"
              label="Reference image"
              hint="The known person, compared against every image below"
              file={referenceFile}
              onChange={setReferenceFile}
            />

            <label className="wk-drop" htmlFor="batch-probes">
              <input
                id="batch-probes"
                type="file"
                accept={ACCEPTED}
                multiple
                aria-label="Images to compare against the reference"
                onChange={(event) => {
                  setResult(null);
                  setError("");
                  setProbeFiles(Array.from(event.target.files || []));
                }}
              />
              <strong>
                {probeFiles.length > 0
                  ? `${probeFiles.length} image${probeFiles.length === 1 ? "" : "s"} selected`
                  : "Images to compare"}
              </strong>
              <small>
                {overLimit
                  ? "Over the 50-image limit — split the batch"
                  : "Up to 50 images, each compared against the reference"}
              </small>
            </label>
          </div>

          <div style={{ height: 20 }} />

          <div className="wk-grid two">
            <label className="wk-field">
              <span>Case</span>
              <select value={caseId} onChange={(event) => setCaseId(event.target.value)}>
                <option value="">No case</option>
                {cases.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.reference} — {item.title}
                  </option>
                ))}
              </select>
            </label>

            <label className="wk-field">
              <span className="wk-required">Lawful basis</span>
              <input
                required
                maxLength={500}
                value={lawfulBasis}
                onChange={(event) => setLawfulBasis(event.target.value)}
                placeholder="Warrant 2026/114 — suspect identification"
              />
            </label>
          </div>

          <button
            type="submit"
            className="wk-button"
            disabled={running || !referenceFile || probeFiles.length === 0 || overLimit}
          >
            {running
              ? `Comparing ${probeFiles.length} image${probeFiles.length === 1 ? "" : "s"}…`
              : "Run comparison"}
          </button>
        </form>
      </section>

      {result && (
        <section className="wk-card">
          <h2>
            Results — {result.succeeded} of {result.submitted} compared
            {result.failed > 0 && `, ${result.failed} failed`}
          </h2>

          {result.reference?.deepfake?.flagged && (
            <div className="wk-banner critical" style={{ marginBottom: 16 }}>
              <div>
                <strong>Reference image: synthetic media suspected</strong>
                The synthetic-media screen flagged the reference image (risk{" "}
                {Math.round((result.reference.deepfake.score || 0) * 100)}%,{" "}
                {result.reference.deepfake.band}). Every comparison below was made against
                suspected generated or manipulated media and cannot support any identity
                conclusion until the reference's provenance is resolved.
              </div>
            </div>
          )}
          {!result.reference?.deepfake?.flagged && result.reference?.deepfake?.review_advised && (
            <div className="wk-banner warn" style={{ marginBottom: 16 }}>
              <div>
                <strong>Reference image: synthetic-media review advised</strong>
                The screen found indicators on the reference image that fall short of a flag
                (risk {Math.round((result.reference.deepfake.score || 0) * 100)}%). Examine the
                reference for manipulation before relying on the comparisons below.
              </div>
            </div>
          )}
          <p style={{ fontSize: 14, lineHeight: 1.6 }}>
            Decision threshold {result.threshold}. Similarity is shown to four decimal places
            so it is visible how close each decision fell to the line.
          </p>

          <div className="wk-table-wrap">
            <table className="wk-table">
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Status</th>
                  <th>Similarity</th>
                  <th>Decision</th>
                  <th>Quality</th>
                  <th>Integrity</th>
                  <th>Audit hash</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((row) => (
                  <tr key={row.index}>
                    <td>
                      <b>{row.label}</b>
                    </td>
                    <td>
                      {row.status === "ok" ? (
                        "OK"
                      ) : (
                        <span title={row.error}>Failed — {row.error}</span>
                      )}
                    </td>
                    <td>{row.similarity != null ? row.similarity.toFixed(4) : "—"}</td>
                    <td>
                      {row.verified == null ? (
                        "—"
                      ) : (
                        <span className={`wk-chip ${row.verified ? "good" : "neutral"}`}>
                          {row.verified ? "Match" : "No match"}
                        </span>
                      )}
                    </td>
                    <td>
                      {row.probe_quality != null ? `${Math.round(row.probe_quality * 100)}%` : "—"}
                    </td>
                    <td>
                      <IntegrityChip band={row.probe_deepfake_band} flags={row.probe_flags} />
                    </td>
                    <td>
                      <code style={{ fontSize: 11 }}>
                        {row.audit_hash ? `${row.audit_hash.slice(0, 12)}…` : "—"}
                      </code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {result.notice && <p className="wk-notice">{result.notice}</p>}
        </section>
      )}

      <section className="wk-card">
        <h2>Forensic report generator</h2>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <button
            type="button"
            className="wk-button"
            disabled={!caseId || Boolean(generatingReport)}
            onClick={() => generateReport("examination")}
          >
            {generatingReport === "examination" ? "Generating…" : "Examination report (PDF)"}
          </button>
          <button
            type="button"
            className="wk-button ghost"
            disabled={!caseId || Boolean(generatingReport)}
            onClick={() => generateReport("pdf")}
          >
            {generatingReport === "pdf" ? "Generating…" : "Case log (PDF)"}
          </button>
        </div>
        {!caseId && (
          <p className="wk-notice" style={{ marginTop: 12 }}>
            Select a case above (and run the comparison under it) to generate its report.
          </p>
        )}
      </section>
    </>
  );
}

/**
 * Per-row verdict of the synthetic-media screen. The chip carries the band;
 * the tooltip lists every integrity flag the image raised, so a row can be
 * triaged without opening it.
 */
function IntegrityChip({ band, flags }) {
  if (!band) return "—";
  const title = flags && flags.length > 0 ? flags.join(", ") : undefined;
  if (band === "high") {
    return (
      <span className="wk-chip bad" title={title}>
        Flagged
      </span>
    );
  }
  if (band === "elevated") {
    return (
      <span className="wk-chip review" title={title}>
        Review
      </span>
    );
  }
  return (
    <span className="wk-chip neutral" title={title}>
      Clear
    </span>
  );
}
