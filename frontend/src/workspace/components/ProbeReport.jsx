const PERCENT = (value) => `${Math.round((Number(value) || 0) * 100)}%`;

const REASON_TEXT = {
  face_too_small: "The face occupies too few pixels to carry reliable identity detail.",
  brightness_out_of_range: "Exposure is outside the usable range.",
  low_contrast: "Contrast is too flat to separate facial structure.",
  blur_risk: "The image is blurred; fine detail the model relies on is missing.",
  yaw_out_of_range: "The head is turned too far from the camera.",
  pitch_out_of_range: "The head is tilted too far up or down.",
  roll_out_of_range: "The head is rotated too far in-plane.",
  detection_confidence_below_minimum: "The detector was not confident this is a face.",
  liveness_below_threshold: "Passive liveness screening flagged this image.",
  low_skin_texture_detail: "Skin texture is unusually smooth, consistent with a print or re-capture.",
  possible_screen_replay_moire: "Periodic interference consistent with a photograph of a screen.",
  flat_colour_distribution: "Colour distribution is unusually flat.",
  synthetic_media_risk: "Frequency artefacts consistent with generated or heavily retouched media.",
  multiple_faces_detected: "More than one face is present; the largest was used.",
  no_landmark_alignment: "No landmarks available, so alignment fell back to a bounding-box crop.",
  recognition_model_unavailable: "No recognition model is loaded; this result carries no meaning.",
  score_in_review_band: "The score sits between the review and match thresholds.",
  low_margin_over_runner_up: "The top candidate barely outscores the next one.",
  empty_gallery: "Nothing is enrolled to compare against.",
};

function scoreClass(value) {
  if (value >= 0.6) return "";
  if (value >= 0.35) return "review";
  return "low";
}

/**
 * Shows why a probe was accepted or rejected.
 *
 * An examiner needs to distinguish "no one in the gallery resembles this
 * person" from "this image was never good enough to search", because those two
 * outcomes mean completely different things for an investigation.
 */
export function ProbeReport({ probe, reasons = [] }) {
  if (!probe) return null;

  const { quality, liveness } = probe;

  return (
    <div>
      <div className="wk-metrics">
        <Metric label="Image quality" value={quality.score} note={quality.accepted ? "Accepted" : "Below gate"} />
        <Metric label="Sharpness" value={quality.sharpness} note={`Laplacian ${quality.laplacian_variance}`} />
        <Metric label="Brightness" value={quality.brightness} />
        <Metric label="Face size" raw={`${quality.face_pixels} px`} note="Shorter edge" />
        <Metric
          label="Liveness screen"
          value={liveness.score}
          note={liveness.passed ? "No flag" : "Flagged"}
        />
        <Metric
          label="Synthetic risk"
          value={probe.deepfake_risk}
          note={probe.deepfake_risk >= 0.65 ? "Elevated" : "Normal"}
        />
      </div>

      {/* Collapsed by default: the screening caveat and capture details matter
          for the record, not for the at-a-glance read, so they live behind a
          disclosure instead of repeating on every assessment. */}
      <details style={{ marginTop: 14 }}>
        <summary style={{ cursor: "pointer", fontSize: 13, opacity: 0.7 }}>
          Technical details
        </summary>

        <dl className="wk-detail-list">
          <div>
            <dt>Detector</dt>
            <dd>{probe.detector}</dd>
          </div>
          <div>
            <dt>Faces found</dt>
            <dd>{probe.faces_detected}</dd>
          </div>
          <div>
            <dt>Head pose</dt>
            <dd>
              yaw {probe.pose.yaw}&deg;, pitch {probe.pose.pitch}&deg;, roll {probe.pose.roll}&deg;
            </dd>
          </div>
        </dl>

        <p className="wk-notice">
          Liveness and synthetic-media screening here are heuristics, not certified detection.
          They have not been evaluated against ISO/IEC 30107-3 and will not stop a determined
          attacker. A pass means nothing obvious was wrong, not that the media is authentic.
        </p>
      </details>

      {reasons.length > 0 && (
        <>
          <h3 className="wk-subhead">Flags raised</h3>
          <ul className="wk-reason-list">
            {reasons.map((reason) => (
              <li key={reason}>
                <code>{reason}</code>
                <span>{REASON_TEXT[reason] || "See the engine documentation for this flag."}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function Metric({ label, value, raw, note }) {
  const numeric = Number(value);
  return (
    <div className="wk-metric">
      <span>{label}</span>
      <strong>{raw ?? PERCENT(value)}</strong>
      {raw === undefined && Number.isFinite(numeric) && (
        <div className="wk-score-bar">
          <i className={scoreClass(numeric)} style={{ width: PERCENT(value) }} />
        </div>
      )}
      {note && <small>{note}</small>}
    </div>
  );
}
