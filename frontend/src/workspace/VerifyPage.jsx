import { useEffect, useState } from "react";
import { listCases, runVerification } from "../services/imatchApi";
import { ImageDropZone } from "./components/ImageDropZone";
import { ProbeReport } from "./components/ProbeReport";

/**
 * Translate the raw comparison into the language an investigator acts on.
 *
 * The cosine value and threshold stay visible in a technical footnote — they
 * are what the audit record holds — but the headline states the finding the
 * way it would be spoken: match, no match, or inconclusive.
 */
function describeVerdict(result) {
  const similarity = Number(result.similarity);
  const threshold = Number(result.threshold);
  if (result.verified) {
    return {
      label: "Match",
      chip: "good",
      detail: "Same person indicated",
      summary:
        "The two faces are similar enough for the engine to treat them as the same person. Confirm by visual examination before relying on this.",
    };
  }
  if (result.morphing?.in_morph_band) {
    return {
      label: "Inconclusive",
      chip: "neutral",
      detail: "Expert review needed",
      summary:
        "The score falls just short of a match — in a range where the algorithm alone cannot make the call.",
    };
  }
  const farBelow = similarity < threshold - 0.15;
  return {
    label: "No match",
    chip: "neutral",
    detail: "Different people indicated",
    summary: farBelow
      ? "The faces show no meaningful similarity. These images are unlikely to show the same person."
      : "The similarity falls short of the level needed to treat these as the same person.",
  };
}

/**
 * One-to-one comparison. No gallery is involved and nothing is enrolled, so
 * this is the right tool for "are these two photographs the same person"
 * questions where both images are already in hand.
 */
export function VerifyPage() {
  const [cases, setCases] = useState([]);
  const [referenceFile, setReferenceFile] = useState(null);
  const [probeFile, setProbeFile] = useState(null);
  const [caseId, setCaseId] = useState("");
  const [lawfulBasis, setLawfulBasis] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    listCases().then(setCases).catch(() => setCases([]));
  }, []);

  const [referencePreview, setReferencePreview] = useState("");
  const [probePreview, setProbePreview] = useState("");
  const verdict = result ? describeVerdict(result) : null;

  useEffect(() => {
    if (!referenceFile) return setReferencePreview("");
    const url = URL.createObjectURL(referenceFile);
    setReferencePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [referenceFile]);

  useEffect(() => {
    if (!probeFile) return setProbePreview("");
    const url = URL.createObjectURL(probeFile);
    setProbePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [probeFile]);

  async function handleSubmit(event) {
    event.preventDefault();
    setRunning(true);
    setError("");
    setResult(null);
    try {
      setResult(await runVerification({ referenceFile, probeFile, caseId, lawfulBasis }));
    } catch (verifyError) {
      setError(verifyError.message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      <div className="wk-page-head">
        <div>
          <h1>1:1 comparison</h1>
          <p>
            Compare two images directly. Neither is stored in the gallery and neither becomes
            searchable.
          </p>
        </div>
      </div>

      {error && <div className="wk-error">{error}</div>}

      <section className="wk-card">
        <form onSubmit={handleSubmit}>
          <div className="wk-grid two">
            <ImageDropZone
              id="reference-image"
              label="Reference image"
              hint="The known person"
              file={referenceFile}
              onChange={setReferenceFile}
            />
            <ImageDropZone
              id="probe-image-verify"
              label="Questioned image"
              hint="The image being tested"
              file={probeFile}
              onChange={setProbeFile}
            />
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
                placeholder="Warrant 2026/114 — document verification"
              />
            </label>
          </div>

          <button type="submit" className="wk-button" disabled={running || !referenceFile || !probeFile}>
            {running ? "Comparing…" : "Compare images"}
          </button>
        </form>
      </section>

      {result && (
        <>
          <section className="wk-card">
            <h2>Comparison</h2>

            <div className="wk-compare">
              <figure>
                {referencePreview && <img src={referencePreview} alt="Reference" />}
                <figcaption>Reference</figcaption>
              </figure>

              <div className="wk-compare-score">
                <b>{verdict.label}</b>
                <span>{verdict.summary}</span>
                <div style={{ marginTop: 12 }}>
                  <span className={`wk-chip ${verdict.chip}`}>{verdict.detail}</span>
                </div>
                <small style={{ display: "block", marginTop: 12, opacity: 0.65 }}>
                  Technical record: cosine similarity {result.similarity.toFixed(4)}, decision
                  threshold {result.threshold.toFixed(4)}
                </small>
              </div>

              <figure>
                {probePreview && <img src={probePreview} alt="Questioned" />}
                <figcaption>Questioned</figcaption>
              </figure>
            </div>

            <p style={{ fontSize: 15, lineHeight: 1.65, marginTop: 22 }}>{result.explanation}</p>

            {result.morphing?.in_morph_band && (
              <div className="wk-banner warn" style={{ marginTop: 16 }}>
                <div>
                  <strong>Possible morphing indicator</strong>
                  The similarity ({result.morphing.similarity.toFixed(4)}) falls in the band
                  where face-morphing attacks typically land: close enough to verify, but short
                  of a genuine same-person pair. Treat this as a prompt to examine the questioned
                  image for blending artefacts, not as a finding.
                </div>
              </div>
            )}

            <p className="wk-notice">{result.notice}</p>
          </section>

          <div className="wk-grid two">
            <section className="wk-card">
              <h2>Reference assessment</h2>
              <ProbeReport probe={result.reference} />
            </section>
            <section className="wk-card">
              <h2>Questioned assessment</h2>
              <ProbeReport probe={result.probe} />
            </section>
          </div>
        </>
      )}
    </>
  );
}
