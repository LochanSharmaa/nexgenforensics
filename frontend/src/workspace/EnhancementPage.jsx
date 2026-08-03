import { useEffect, useRef, useState } from "react";
import {
  analyzeForEnhancement,
  fetchEnhancementImage,
  fileToBase64,
  listCases,
  recogniseEnhancement,
  runEnhancement,
} from "../services/imatchApi";
import { CandidateTable } from "./components/CandidateTable";
import { ImageDropZone } from "./components/ImageDropZone";
import { MetricDelta } from "./components/MetricDelta";
import { PipelinePlan } from "./components/PipelinePlan";
import { SplitCompare } from "./components/SplitCompare";

const DECISION_LABEL = {
  candidate_match: "Candidate match — examiner verification required",
  review_required: "Review required",
  no_match: "No match",
  probe_rejected: "Probe rejected",
  no_face_detected: "No face detected",
  recognition_unavailable: "Recognition unavailable",
  unavailable: "Image unavailable",
};

/**
 * Forensic image enhancement.
 *
 * The page's layout encodes the evidential rule: the original is always on
 * screen next to the enhanced result, the enhanced result always carries its
 * track label, and the A/B recognition view marks the original as primary.
 */
export function EnhancementPage() {
  const [cases, setCases] = useState([]);
  const [file, setFile] = useState(null);
  const [imageBase64, setImageBase64] = useState(null);
  const [caseId, setCaseId] = useState("");
  const [lawfulBasis, setLawfulBasis] = useState("");

  const [analysis, setAnalysis] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  const [result, setResult] = useState(null);
  const [originalUrl, setOriginalUrl] = useState("");
  const [enhancedUrl, setEnhancedUrl] = useState("");

  const [recognising, setRecognising] = useState(false);
  const [recognition, setRecognition] = useState(null);
  const urlsRef = useRef([]);

  useEffect(() => {
    listCases().then(setCases).catch(() => setCases([]));
    return () => urlsRef.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  // Analyse on selection. The endpoint is measurement-only — nothing is stored
  // and nothing is audited — so calling it per upload is free of consequences.
  useEffect(() => {
    if (!file) {
      setAnalysis(null);
      setImageBase64(null);
      return;
    }
    let cancelled = false;
    setAnalyzing(true);
    setError("");
    (async () => {
      try {
        const b64 = await fileToBase64(file);
        if (cancelled) return;
        setImageBase64(b64);
        const report = await analyzeForEnhancement({ imageBase64: b64 });
        if (!cancelled) setAnalysis(report);
      } catch (analysisError) {
        if (!cancelled) setError(analysisError.message);
      } finally {
        if (!cancelled) setAnalyzing(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [file]);

  async function handleEnhance(event) {
    event.preventDefault();
    if (!imageBase64) return;
    setRunning(true);
    setError("");
    setResult(null);
    setRecognition(null);
    setOriginalUrl("");
    setEnhancedUrl("");

    try {
      const outcome = await runEnhancement({
        imageBase64,
        caseId: caseId || null,
        lawfulBasis,
      });
      setResult(outcome);
      const [orig, enh] = await Promise.all([
        fetchEnhancementImage(outcome.enhancement_id, "original"),
        fetchEnhancementImage(outcome.enhancement_id, "enhanced"),
      ]);
      urlsRef.current.push(orig, enh);
      setOriginalUrl(orig);
      setEnhancedUrl(enh);
    } catch (enhanceError) {
      setError(enhanceError.message);
    } finally {
      setRunning(false);
    }
  }

  async function handleRecognise() {
    if (!result) return;
    setRecognising(true);
    setError("");
    try {
      setRecognition(
        await recogniseEnhancement(result.enhancement_id, { lawfulBasis, topK: 10 }),
      );
    } catch (recogniseError) {
      setError(recogniseError.message);
    } finally {
      setRecognising(false);
    }
  }

  const profile = analysis?.profile;

  return (
    <>
      <div className="wk-page-head">
        <div>
          <h1>Image enhancement</h1>
          <p>
            Improve a poor-quality CCTV or surveillance image for visual examination. The original
            is preserved untouched and every processing step is measured, justified, and recorded
            in the audit chain.
          </p>
        </div>
      </div>

      {error && <div className="wk-error">{error}</div>}

      <div className="wk-grid two">
        <section className="wk-card">
          <h2>Source image</h2>
          <p>CCTV frames, doorbell and body-worn captures, dashcam stills, phone recordings.</p>

          <form onSubmit={handleEnhance}>
            <ImageDropZone
              id="enhance-image"
              label="Drop or select the image to enhance"
              file={file}
              onChange={setFile}
            />

            <div style={{ height: 18 }} />

            <label className="wk-field">
              <span>Case</span>
              <select value={caseId} onChange={(event) => setCaseId(event.target.value)}>
                <option value="">No case (ad hoc)</option>
                {cases.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.reference} — {item.title}
                  </option>
                ))}
              </select>
            </label>

            <label className="wk-field">
              <span>Lawful basis</span>
              <input
                maxLength={500}
                value={lawfulBasis}
                onChange={(event) => setLawfulBasis(event.target.value)}
                placeholder="Warrant 2026/114 — CCTV review"
              />
              <small>Recorded verbatim in the audit chain alongside the processing record.</small>
            </label>

            <button className="wk-button" type="submit" disabled={!file || running || analyzing}>
              {running ? "Enhancing…" : "Enhance image"}
            </button>
          </form>
        </section>

        <section className="wk-card">
          <h2>Degradation analysis</h2>
          {!file && <p className="wk-muted">Select an image to see what the system measures in it.</p>}
          {analyzing && <p className="wk-muted">Measuring…</p>}
          {profile && (
            <>
              <div className="wk-stat-row">
                <div className="wk-stat">
                  <b>{profile.width}×{profile.height}</b>
                  <small>stored size</small>
                </div>
                <div className="wk-stat">
                  <b>{Math.round(profile.effective_resolution_ratio * 100)}%</b>
                  <small>true resolution</small>
                </div>
                <div className="wk-stat">
                  <b>{profile.jpeg_quality ?? "—"}</b>
                  <small>est. JPEG quality</small>
                </div>
                <div className="wk-stat">
                  <b>{profile.blur_kind}</b>
                  <small>blur type</small>
                </div>
              </div>

              {analysis.notes.length > 0 && (
                <ul className="wk-notes">
                  {analysis.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              )}

              <h3>Planned pipeline</h3>
              <PipelinePlan plan={analysis.recommended_plan} stages={[]} />
            </>
          )}
        </section>
      </div>

      {result && (
        <section className="wk-card">
          <div className="wk-result-head">
            <h2>Comparison</h2>
            <span className={`wk-chip ${result.track === "reconstructed" ? "review" : "good"}`}>
              {result.label}
            </span>
          </div>

          <SplitCompare
            originalUrl={originalUrl}
            enhancedUrl={enhancedUrl}
            enhancedLabel={
              result.track === "reconstructed"
                ? "AI-ENHANCED PREVIEW — not evidentiary"
                : "PROCESSED — deterministic only"
            }
          />

          {result.warnings?.length > 0 && (
            <div className="wk-warnings">
              {result.warnings.map((warning) => (
                <div key={warning} className="wk-warning">{warning}</div>
              ))}
            </div>
          )}

          <div className="wk-grid two">
            <div>
              <h3>Quality metrics</h3>
              <MetricDelta before={result.metrics_before} after={result.metrics_after} />
              <small className="wk-muted">
                Quality metrics measure legibility, not identity. An improvement here does not make
                any match more reliable.
              </small>
            </div>
            <div>
              <h3>Processing record</h3>
              <PipelinePlan plan={result.plan} stages={result.stages} />
              <small className="wk-muted">
                {result.device.toUpperCase()} · {result.total_ms} ms
                {result.served_from_cache ? " · served from cache" : ""} · original SHA-256{" "}
                <code>{result.original_sha256.slice(0, 16)}…</code>
              </small>
            </div>
          </div>

          <div className="wk-recognise-bar">
            <button
              type="button"
              className="wk-button"
              onClick={handleRecognise}
              disabled={recognising}
            >
              {recognising ? "Searching…" : "Run recognition A/B (original vs enhanced)"}
            </button>
            <small>
              Both searches use the unmodified recogniser. The original&apos;s result is primary;
              the enhanced result is for comparison only.
            </small>
          </div>
        </section>
      )}

      {recognition && (
        <section className="wk-card">
          <h2>Recognition comparison</h2>
          <div className="wk-caution">{recognition.caution}</div>

          <div className="wk-grid two">
            {["original", "enhanced"].map((side) => {
              const data = recognition[side];
              return (
                <div key={side} className={side === "original" ? "wk-primary-side" : ""}>
                  <h3>
                    {side === "original" ? "Original (primary)" : `Enhanced (${data.source_kind})`}
                  </h3>
                  <p className="wk-decision">
                    {DECISION_LABEL[data.decision] || data.decision}
                    {data.explanation ? ` — ${data.explanation}` : ""}
                  </p>
                  {data.candidates?.length > 0 ? (
                    <CandidateTable candidates={data.candidates} readOnly={side === "enhanced"} />
                  ) : (
                    <p className="wk-muted">No candidates returned.</p>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}
    </>
  );
}
