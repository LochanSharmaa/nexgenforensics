import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { listCases, listCandidates, runSearch } from "../services/imatchApi";
import { CandidateTable } from "./components/CandidateTable";
import { ImageDropZone } from "./components/ImageDropZone";
import { ProbeReport } from "./components/ProbeReport";

const DECISION_TONE = {
  candidate_match: "good",
  review_required: "review",
  no_match: "neutral",
  probe_rejected: "bad",
  recognition_unavailable: "bad",
};

const DECISION_LABEL = {
  candidate_match: "Candidate match — examiner verification required",
  review_required: "Review required",
  no_match: "No match",
  probe_rejected: "Probe rejected",
  recognition_unavailable: "Recognition unavailable",
};

export function SearchPage() {
  const [params] = useSearchParams();
  const [cases, setCases] = useState([]);
  const [file, setFile] = useState(null);
  const [caseId, setCaseId] = useState(params.get("case") || "");
  const [lawfulBasis, setLawfulBasis] = useState("");
  const [purpose, setPurpose] = useState("");
  const [topK, setTopK] = useState(10);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [candidates, setCandidates] = useState([]);

  useEffect(() => {
    listCases().then(setCases).catch(() => setCases([]));
  }, []);

  async function refreshCandidates(searchId) {
    try {
      setCandidates(await listCandidates(searchId));
    } catch {
      /* keep whatever the search already returned */
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setRunning(true);
    setError("");
    setResult(null);
    setCandidates([]);

    try {
      const response = await runSearch({ file, caseId, lawfulBasis, purpose, topK });
      setResult(response);
      setCandidates(response.candidates);
    } catch (searchError) {
      setError(searchError.message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      <div className="wk-page-head">
        <div>
          <h1>Face search</h1>
          <p>
            Compare a probe image against the subjects your organisation has enrolled. Results
            are ranked by similarity and require examiner verification before use.
          </p>
        </div>
      </div>

      {error && <div className="wk-error">{error}</div>}

      <div className="wk-grid two">
        <section className="wk-card">
          <h2>Probe</h2>
          <p>One face per image. The largest detected face is used.</p>

          <form onSubmit={handleSubmit}>
            <ImageDropZone
              id="probe-image"
              label="Drop or select the probe image"
              file={file}
              onChange={setFile}
            />

            <div style={{ height: 18 }} />

            <label className="wk-field">
              <span>Case</span>
              <select value={caseId} onChange={(event) => setCaseId(event.target.value)}>
                <option value="">No case (ad hoc search)</option>
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
              <small>
                Recorded verbatim in the audit chain. The system cannot judge whether a search
                is lawful; it can only make sure a reason was stated and preserved.
              </small>
            </label>

            <label className="wk-field">
              <span>Purpose</span>
              <input
                maxLength={500}
                value={purpose}
                onChange={(event) => setPurpose(event.target.value)}
                placeholder="Identify subject from CCTV still, incident 2026-0114-A"
              />
            </label>

            <label className="wk-field">
              <span>Candidates to return</span>
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </label>

            <button type="submit" className="wk-button" disabled={running || !file}>
              {running ? "Searching…" : "Run search"}
            </button>
          </form>
        </section>

        <section className="wk-card">
          <h2>Probe assessment</h2>
          <p>Whether this image was good enough to search, and why.</p>

          {result ? (
            <ProbeReport probe={result.probe} reasons={result.reasons} />
          ) : (
            <div className="wk-empty">Run a search to see the probe assessment.</div>
          )}
        </section>
      </div>

      {result && (
        <section className="wk-card">
          <h2>Result</h2>

          <div style={{ marginBottom: 14 }}>
            <span className={`wk-chip ${DECISION_TONE[result.decision] || "neutral"}`}>
              {DECISION_LABEL[result.decision] || result.decision}
            </span>
          </div>

          <p style={{ fontSize: 15, lineHeight: 1.65, margin: "0 0 16px" }}>{result.explanation}</p>

          <div className="wk-metrics" style={{ marginBottom: 20 }}>
            <div className="wk-metric">
              <span>Top similarity</span>
              <strong>{result.confidence.toFixed(4)}</strong>
              <small>Cosine, not a probability</small>
            </div>
            <div className="wk-metric">
              <span>Margin over runner-up</span>
              <strong>{result.margin.toFixed(4)}</strong>
              <small>Small margin = weak lead</small>
            </div>
            <div className="wk-metric">
              <span>Gallery searched</span>
              <strong>{result.gallery_size}</strong>
              <small>templates</small>
            </div>
            <div className="wk-metric">
              <span>Elapsed</span>
              <strong>{result.duration_ms} ms</strong>
            </div>
          </div>

          <CandidateTable
            candidates={candidates}
            thresholds={result.thresholds}
            onUpdated={() => refreshCandidates(result.search_id)}
          />

          <p className="wk-notice">{result.notice}</p>

          <p style={{ fontSize: 12.5, color: "var(--muted)" }}>
            Search <span className="wk-mono">{result.search_id}</span> · audit entry{" "}
            <span className="wk-mono">{result.audit_hash?.slice(0, 16)}…</span> · model{" "}
            {result.model.backend} ({result.model.model_pack})
          </p>
        </section>
      )}

    </>
  );
}
