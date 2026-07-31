import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { normalizeVerifyResult, runVerification } from "../../services/imatchApi";
import "../sections/FaceSearchExperience.css";
import "./IndividualComparePage.css";

/**
 * The "Individual" destination: 1:1 comparison and nothing else.
 *
 * Deliberately absent: cases, 1:N search, batch, enrolment, audit-trail UI.
 * Someone who chose this mode wants to compare two photographs.
 *
 * It reuses FaceSearchExperience.css rather than restyling, so the cards,
 * drop zones, score bar and verdict block are literally the same components
 * the marketing demo uses — same fonts, colours and card treatment by
 * construction rather than by imitation.
 *
 * ON THE "REASON" FIELD
 * ---------------------
 * The API requires a lawful basis and writes it verbatim into the audit chain.
 * It is kept here even though this is the stripped-down page, because the two
 * ways to remove it are both worse: dropping it makes every request fail, and
 * auto-filling it puts a sentence nobody wrote into a permanent audit record.
 * The audit trail is not SHOWN on this page; that is a different thing from
 * not being written.
 */

const LABELS = {
  same_person: { text: "Same person", color: "#1f7a4d", bg: "rgba(31,122,77,0.12)", icon: "✓" },
  different_person: { text: "Different person", color: "#9a2f42", bg: "rgba(154,47,66,0.12)", icon: "✕" },
};

function DropZone({ id, label, preview, onChange }) {
  return (
    <label className="im-compare-drop" htmlFor={id}>
      <input
        id={id}
        type="file"
        accept="image/*"
        hidden
        onChange={(event) => onChange(event.target.files?.[0] || null)}
      />
      {preview ? (
        <img src={preview} alt={`${label} preview`} />
      ) : (
        <span className="im-compare-drop-empty">
          <b>{label}</b>
          <small>Click to upload</small>
        </span>
      )}
    </label>
  );
}

export function IndividualComparePage() {
  const [refFile, setRefFile] = useState(null);
  const [probeFile, setProbeFile] = useState(null);
  const [refPreview, setRefPreview] = useState("");
  const [probePreview, setProbePreview] = useState("");
  const [reason, setReason] = useState("");
  const [runState, setRunState] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!refFile) { setRefPreview(""); return undefined; }
    const url = URL.createObjectURL(refFile);
    setRefPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [refFile]);

  useEffect(() => {
    if (!probeFile) { setProbePreview(""); return undefined; }
    const url = URL.createObjectURL(probeFile);
    setProbePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [probeFile]);

  async function handleRun() {
    setError("");
    setResult(null);

    if (!reason.trim()) {
      setError("Give a reason for this comparison before running it.");
      setRunState("error");
      return;
    }

    setRunState("running");
    try {
      const n = normalizeVerifyResult(
        await runVerification({
          referenceFile: refFile,
          probeFile,
          lawfulBasis: reason.trim(),
        }),
      );
      setResult({ ...n, label: n.verified ? "same_person" : "different_person" });
      setRunState("complete");
    } catch (runError) {
      setError(runError.message);
      setRunState("error");
    }
  }

  const cfg = result ? LABELS[result.label] : null;
  const scorePct = result ? `${(result.similarity * 100).toFixed(1)}%` : "—";
  const reviewRequired =
    result && (!result.reference.qualityAccepted || !result.probe.qualityAccepted);

  return (
    <section className="nx-solo-page" id="top">
      <div className="nx-solo-head">
        <p className="nx-kicker">iMATCH</p>
        <h1>Compare two photographs</h1>
        <p className="nx-solo-sub">
          Upload a reference image and a probe image. The comparison runs on the same engine and
          the same decision threshold used throughout iMATCH.
        </p>
      </div>

      <div className="nx-solo-card">
        <div className="im-compare-uploads">
          <DropZone
            id="solo-ref"
            label="Reference image"
            preview={refPreview}
            onChange={(f) => { setRefFile(f); setResult(null); setError(""); }}
          />
          <div className="im-compare-vs" aria-hidden="true">VS</div>
          <DropZone
            id="solo-probe"
            label="Probe image"
            preview={probePreview}
            onChange={(f) => { setProbeFile(f); setResult(null); setError(""); }}
          />
        </div>

        <label className="im-lawful-field">
          <span>Reason for this comparison (required, recorded against your account)</span>
          <input
            type="text"
            value={reason}
            maxLength={500}
            placeholder="e.g. Confirming a photo of the same person"
            onChange={(event) => setReason(event.target.value)}
          />
        </label>

        {result && (
          <div className="im-compare-result" aria-live="polite" style={{ borderColor: cfg.color }}>
            <div className="im-compare-verdict" style={{ background: cfg.bg, color: cfg.color }}>
              <span className="im-verdict-icon">{cfg.icon}</span>
              <span className="im-verdict-text">{cfg.text}</span>
              <span className="im-verdict-score">{scorePct} similarity</span>
            </div>
            <div className="im-compare-meta">
              <div>
                <b>Decision threshold</b>
                <span>
                  {typeof result.threshold === "number"
                    ? `similarity ≥ ${result.threshold.toFixed(2)} → supports same person`
                    : "reported by engine per comparison"}
                </span>
              </div>
              {reviewRequired && (
                <div className="im-compare-review">
                  ⚠ Image quality is below the accepted band — treat this result with caution
                </div>
              )}
            </div>
          </div>
        )}

        {error && <p className="im-error" role="alert">{error}</p>}

        <button
          type="button"
          className="im-launch"
          onClick={handleRun}
          disabled={runState === "running" || !refFile || !probeFile}
        >
          {runState === "running" ? "Comparing…" : result ? "Run again" : "Run comparison"}
        </button>

        <p className="nx-solo-foot">
          Need cases, 1:N search or batch processing? <Link to="/workspace">Open the full workspace</Link>
        </p>
      </div>
    </section>
  );
}
