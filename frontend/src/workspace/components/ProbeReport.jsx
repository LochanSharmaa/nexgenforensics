const PERCENT = (value) => `${Math.round((Number(value) || 0) * 100)}%`;

const REASON_TEXT = {
  face_too_small: "The face occupies too few pixels to carry reliable identity detail.",
  brightness_out_of_range: "Exposure is outside the usable range.",
  low_contrast: "Contrast is too flat to separate facial structure.",
  blur_risk: "The image is blurred; fine detail the model relies on is missing.",
  severe_blur:
    "High-frequency detail is almost entirely absent, so the image cannot support a reliable comparison.",
  yaw_out_of_range: "The head is turned too far from the camera.",
  pitch_out_of_range: "The head is tilted too far up or down.",
  roll_out_of_range: "The head is rotated too far in-plane.",
  detection_confidence_below_minimum: "The detector was not confident this is a face.",
  liveness_below_threshold: "Passive liveness screening flagged this image.",
  low_skin_texture_detail: "Skin texture is unusually smooth, consistent with a print or re-capture.",
  possible_screen_replay_moire: "Periodic interference consistent with a photograph of a screen.",
  flat_colour_distribution: "Colour distribution is unusually flat.",
  synthetic_media_risk:
    "The synthetic-media screen flagged this image. Treat it as suspected generated or manipulated media until its provenance is resolved.",
  synthetic_media_review_advised:
    "The synthetic-media screen found indicators short of a flag. Examine the image for manipulation before relying on results.",
  generative_metadata_present:
    "The file's own metadata names an AI generation tool or carries a generation record. This is close to conclusive.",
  ai_retouch_metadata_present:
    "Metadata names an AI face-retouching application; the face no longer shows what the camera captured.",
  periodic_upsampling_artefacts:
    "The frequency spectrum repeats periodically — the fingerprint of generator upsampling or naive rescaling.",
  spectral_energy_anomaly:
    "The frequency-energy distribution does not match how lenses and sensors form photographs.",
  synthetic_texture_smoothness:
    "Mid-frequency texture has collapsed relative to coarse structure, typical of generated or heavily retouched skin.",
  sensor_noise_absent:
    "No sensor noise where a camera would leave it — consistent with generation, heavy denoising, or aggressive compression.",
  noise_pattern_inconsistent: "Noise levels vary across the image more than one capture can explain.",
  face_background_noise_mismatch:
    "The face region carries different noise than the background it sits in — the signature of a pasted or generated face.",
  ela_region_anomaly:
    "The face region responds to JPEG recompression differently from the rest of the image, indicating a separate compression history.",
  face_boundary_blending_artefacts:
    "The transition ring around the face is unnaturally smooth, consistent with a blended face swap.",
  multiple_faces_detected: "More than one face is present; the largest was used.",
  no_landmark_alignment: "No landmarks available, so alignment fell back to a bounding-box crop.",
  recognition_model_unavailable: "No recognition model is loaded; this result carries no meaning.",
  score_in_review_band: "The score sits between the review and match thresholds.",
  low_margin_over_runner_up: "The top candidate barely outscores the next one.",
  empty_gallery: "Nothing is enrolled to compare against.",
};

const SIGNAL_LABEL = {
  provenance: "Provenance metadata",
  spectral_slope: "Spectral slope",
  spectral_peaks: "Spectral periodicity",
  texture: "Texture energy",
  noise_floor: "Sensor-noise floor",
  noise_consistency: "Noise consistency",
  ela: "Recompression (ELA)",
  boundary: "Face boundary",
};

const BAND_PRESENTATION = {
  high: { label: "Flagged", chip: "bad" },
  elevated: { label: "Review advised", chip: "review" },
  moderate: { label: "Weak indicators", chip: "neutral" },
  minimal: { label: "No indicators", chip: "good" },
};

function scoreClass(value) {
  if (value >= 0.6) return "";
  if (value >= 0.35) return "review";
  return "low";
}

// Risk bars invert the palette: a HIGH value is the bad outcome.
function riskClass(value) {
  if (value >= 0.6) return "low";
  if (value >= 0.35) return "review";
  return "";
}

/**
 * Shows why a probe was accepted or rejected.
 *
 * An examiner needs to distinguish "no one in the gallery resembles this
 * person" from "this image was never good enough to search", because those two
 * outcomes mean completely different things for an investigation. The
 * synthetic-media screen adds a third outcome that overrides both: "this may
 * not be a photograph of a real scene at all".
 */
export function ProbeReport({ probe, reasons }) {
  if (!probe) return null;

  const { quality, liveness, deepfake } = probe;
  const flagList = reasons ?? probe.reasons ?? [];
  const band = BAND_PRESENTATION[deepfake?.band] || null;
  const signals = (deepfake?.signals || []).filter((signal) => signal.weight > 0);

  return (
    <div>
      <div className="wk-metrics">
        <Metric label="Image quality" value={quality.score} note={quality.accepted ? "Accepted" : "Below gate"} />
        <Metric
          label="Sharpness"
          value={quality.sharpness}
          note={`Laplacian ${quality.laplacian_variance} at 112 px`}
        />
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
          risk
          note={band ? band.label : probe.deepfake_risk >= 0.65 ? "Elevated" : "Normal"}
        />
      </div>

      {deepfake && signals.length > 0 && (
        <>
          <h3 className="wk-subhead">
            Synthetic-media screen{" "}
            {band && <span className={`wk-chip ${band.chip}`}>{band.label}</span>}
          </h3>
          <ul className="wk-reason-list">
            {signals.map((signal) => (
              <li key={signal.name}>
                <code>
                  {SIGNAL_LABEL[signal.name] || signal.name} {PERCENT(signal.score)}
                </code>
                <span>{signal.detail}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      {flagList.length > 0 && (
        <>
          <h3 className="wk-subhead">Flags raised</h3>
          <ul className="wk-reason-list">
            {flagList.map((reason) => (
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

function Metric({ label, value, raw, note, risk = false }) {
  const numeric = Number(value);
  return (
    <div className="wk-metric">
      <span>{label}</span>
      <strong>{raw ?? PERCENT(value)}</strong>
      {raw === undefined && Number.isFinite(numeric) && (
        <div className="wk-score-bar">
          <i
            className={risk ? riskClass(numeric) : scoreClass(numeric)}
            style={{ width: PERCENT(value) }}
          />
        </div>
      )}
      {note && <small>{note}</small>}
    </div>
  );
}
