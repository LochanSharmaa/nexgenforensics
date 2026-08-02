import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./FaceSearchExperience.css";
import faceSearchVideo from "../../assets/facesearch.mp4";
import { useLoginGate } from "../../hooks/useLoginGate";
import {
  imatchApiUrl,
  runVerification,
  normalizeVerifyResult,
  runBatch,
} from "../../services/imatchApi";

// Each item must correspond to something implemented and tested. See CLAIMS.md.
// "99.99% Benchmark Target" was a placeholder with no measurement behind it and
// is replaced by a real, dataset-qualified number from BENCHMARKS.md. Accuracy
// must never be quoted without naming the dataset and the task (1:1
// verification vs. rank-1 identification).
const trustItems = [
  "Commercial Face Matching",
  "Image Quality & Capture Check",
  "Synthetic-Media Artifact Screen",
  "Tenant-Isolated Gallery",
  "99.78% 1:1 verification — LFW, 6,000 pairs",
];

const storySteps = [
  {
    label: "Upload Face",
    title: "Start with an image, URL, or batch intake.",
    body: "Authorized operators submit a face image through the console, paste a storage path, or queue a batch for secure recognition.",
    mode: "upload",
    score: "12%",
    results: ["Upload zone ready", "Consent check required", "Batch intake available"],
  },
  {
    label: "Quality Assessment",
    title: "Measure whether the input is reliable enough to search.",
    body: "iMatch checks blur, lighting, pose, occlusion, and capture quality before any identity result is trusted.",
    mode: "quality",
    score: "48%",
    results: ["Blur: acceptable", "Lighting: strong", "Pose: reviewable"],
  },
  {
    label: "Landmark Detection",
    title: "Map facial structure into recognition-ready signals.",
    body: "A restrained preview overlay marks the 468-point landmark model without turning the page into a noisy dashboard.",
    mode: "landmarks",
    score: "71%",
    results: ["468 landmark points", "Eyes, nose, jawline", "Pose normalized"],
  },
  {
    label: "Face Matching",
    title: "Return ranked candidates with confidence indicators.",
    body: "Similarity vectors compare the submitted face against permitted workspace datasets and surface ranked candidate rows.",
    mode: "matching",
    score: "93%",
    results: ["Candidate A - 96.8%", "Candidate B - 87.1%", "Candidate C - 82.6%"],
  },
  {
    label: "Verification Complete",
    title: "Confirm authenticity before results move forward.",
    body: "Capture-quality and synthetic-media artifact screens are recorded alongside the recognition score, and every search writes a retrievable audit record.",
    mode: "complete",
    score: "96.8%",
    results: ["Quality check passed", "Artifact screen passed", "Audit record written"],
  },
];

const tabs = [
  {
    id: "console",
    label: "Web Console",
    title: "Recognition workflow dashboard",
    lines: ["Single search intake", "Quality score 94/100", "Liveness passed", "Candidate confidence 96.8%"],
  },
  {
    id: "api",
    label: "REST API",
    title: "Premium API request",
    lines: [
      "POST /api/biometrics/verify",
      "{",
      '  "reference": "face-a.jpg",',
      '  "probe": "face-b.jpg",',
      '  "operator_id": "demo_operator"',
      "}",
    ],
  },
  {
    id: "batch",
    label: "Batch Processing",
    title: "File queue and results table",
    lines: ["batch-017.csv queued", "1,248 images normalized", "37 candidates flagged", "audit export ready"],
  },
  {
    id: "secure",
    label: "Secure Integrations",
    title: "Connected enterprise systems",
    lines: ["KYC system", "Access control", "Identity records", "Review workflow"],
  },
];

const searchModes = [
  {
    id: "single",
    label: "Single Search",
    title: "Drop image, paste anywhere, or click to upload",
    formats: "JPG - JPEG - PNG - WEBP - HEIC - TIFF - BMP",
    urlLabel: "Source URL",
    placeholder: "Enter image URL or cloud storage path",
    summary: ["Single subject intake", "Consent-aware search", "One face image"],
  },
  {
    id: "compare",
    label: "1:1 Comparison",
    title: "Upload two face images for biometric comparison",
    formats: "Reference image + probe image",
    urlLabel: "Comparison Sources",
    placeholder: "Enter reference and probe image URLs",
    summary: ["Reference subject", "Probe image", "Verification score"],
  },
  {
    id: "batch",
    label: "Batch Upload",
    title: "Drop a folder or CSV manifest for batch matching",
    formats: "ZIP - CSV - JPG - PNG - WEBP",
    urlLabel: "Batch Manifest",
    placeholder: "Enter manifest URL or cloud batch path",
    summary: ["Batch queue ready", "Multi-image processing", "Result export"],
  },
  // "URL Import" mode removed alongside the URL field: the API accepts
  // multipart uploads only, so this mode could never complete a search.
  // See CLAIMS.md for the SSRF requirements before reinstating it.
];

// ─── Label → display config ──────────────────────────────────────────────────
const LABEL_CONFIG = {
  same_person: { text: "Same Person", color: "#1a7a4a", bg: "rgba(26,122,74,0.10)", icon: "✓" },
  inconclusive: { text: "Inconclusive", color: "#9a6e00", bg: "rgba(154,110,0,0.10)", icon: "~" },
  different_person: { text: "Different Person", color: "#9a2f42", bg: "rgba(154,47,66,0.10)", icon: "✗" },
  unknown: { text: "Unknown", color: "#6f6860", bg: "rgba(111,104,96,0.10)", icon: "?" },
};

export function FaceSearchExperience() {
  const [activeStep, setActiveStep] = useState(0);
  const [activeTab, setActiveTab] = useState("console");
  const stepRefs = useRef([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActiveStep(Number(visible.target.dataset.step));
      },
      { threshold: [0.42, 0.62, 0.82] }
    );

    stepRefs.current.forEach((node) => node && observer.observe(node));
    return () => observer.disconnect();
  }, []);

  const activeTabData = useMemo(
    () => tabs.find((tab) => tab.id === activeTab) ?? tabs[0],
    [activeTab]
  );

  return (
    <>
      <section id="top" className="im-hero im-section" aria-labelledby="imatch-title">
        <div className="im-orb im-orb-one" aria-hidden="true" />
        <div className="im-hero-copy">
          <p className="im-eyebrow">NexGen Identity Product Suite</p>
          {/* Real measured number, dataset-qualified. See BENCHMARKS.md section 2.
              The previous "validation target 99.99%" was an aspiration with no
              measurement behind it, presented as a capability. */}
          <p className="im-badge">
            Enterprise biometric engine - 99.78% 1:1 verification (LFW, 6,000 pairs)
          </p>
          <h1 id="imatch-title">NexGen iMatch</h1>
          <h2>Enterprise Facial Recognition System</h2>
          <p>
            Advanced facial recognition for commercial face search, identity
            verification, fraud prevention, access control, capture-quality
            screening, synthetic-media artifact checks, and audited recognition
            workflows.
          </p>
          <div className="im-hero-actions">
            <a href="#story">Start Face Search</a>
            <a href="#briefing">Request Access</a>
          </div>
          <ul className="im-trust-row" aria-label="iMatch trust signals">
            {trustItems.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <ImatchUploadConsole step={storySteps[0]} hero />
      </section>

      <section id="story" className="im-story im-section" aria-labelledby="story-title">
        <div className="im-section-heading">
          <p className="im-eyebrow">Recognition Workflow</p>
          <h2 id="story-title">From Face Upload to Enterprise Identity Decision</h2>
        </div>
        <div className="im-story-grid">
          <div className="im-story-copy">
            {storySteps.map((step, index) => (
              <article
                className={index === activeStep ? "im-step-card active" : "im-step-card"}
                key={step.label}
                data-step={index}
                ref={(node) => {
                  stepRefs.current[index] = node;
                }}
              >
                <span>{String(index + 1).padStart(2, "0")}</span>
                <h3>{step.label}</h3>
                <h4>{step.title}</h4>
                <p>{step.body}</p>
              </article>
            ))}
          </div>
          <figure className="im-story-face-visual">
            <video
              src={faceSearchVideo}
              autoPlay
              muted
              loop
              playsInline
              aria-label="Facial recognition workflow preview"
            />
          </figure>
        </div>
      </section>

      <section id="research" className="im-interface im-section" aria-labelledby="interface-title">
        <div className="im-section-heading">
          <p className="im-eyebrow">Interfaces</p>
          <h2 id="interface-title">Flexible Interfaces for Every Recognition Workflow</h2>
        </div>
        <div className="im-interface-shell">
          <div className="im-tabs" role="tablist" aria-label="iMatch interfaces">
            {tabs.map((tab) => (
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === tab.id}
                className={activeTab === tab.id ? "active" : ""}
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <article className="im-tab-panel" key={activeTabData.id}>
            <h3>{activeTabData.title}</h3>
            <div className={activeTabData.id === "api" ? "im-code-card" : "im-preview-card"}>
              {activeTabData.lines.map((line) => (
                <span key={line}>{line}</span>
              ))}
            </div>
          </article>
        </div>
      </section>

      <section id="briefing" className="im-final-cta im-section">
        <div className="im-orb im-orb-three" aria-hidden="true" />
        <p className="im-eyebrow">Deploy iMatch</p>
        <h2>Bring AI Facial Recognition Into Your Enterprise Identity Workflow</h2>
        <p>
          Deploy face search, biometric verification, and authenticity checks inside
          a secure, tenant-isolated recognition system with audit-ready controls.
        </p>
        <div className="im-hero-actions">
          <a href="mailto:access@nexgenforensics.ai">Request Access</a>
          <a href="/#platform">Explore Product Suite</a>
        </div>
      </section>
    </>
  );
}

// ─── ImatchUploadConsole ──────────────────────────────────────────────────────
function ImatchUploadConsole({ step, hero = false }) {
  const [activeMode, setActiveMode] = useState(searchModes[0].id);

  return (
    <div className={`im-upload-console ${step.mode}${hero ? " hero" : ""}`} aria-label="iMatch Face Search">
      <div className="im-console-head">
        <div>
          <span>NexGen Identity Recognition Console</span>
          <h3>iMatch Face Search</h3>
        </div>
        <strong>IM-468</strong>
      </div>

      <div className="im-search-tabs" role="tablist" aria-label="Search mode">
        {searchModes.map((tab) => (
          <button
            type="button"
            className={activeMode === tab.id ? "active" : ""}
            role="tab"
            aria-selected={activeMode === tab.id}
            key={tab.id}
            onClick={() => setActiveMode(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeMode === "compare" ? (
        <ComparePanel />
      ) : activeMode === "batch" ? (
        <BatchPanel />
      ) : (
        <SingleSearchPanel step={step} activeMode={activeMode} />
      )}
    </div>
  );
}

// ─── BatchPanel — real 1:N batch search UI ────────────────────────────────────
function BatchPanel() {
  const requireLogin = useLoginGate();
  const [batchFiles, setBatchFiles] = useState([]);
  const [referenceFile, setReferenceFile] = useState(null);
  const [mode, setMode] = useState("one_to_many");
  const [lawfulBasis, setLawfulBasis] = useState("");
  const [runState, setRunState] = useState("idle");
  const [batchResult, setBatchResult] = useState(null);
  const [error, setError] = useState("");

  const handleFilesChange = (e) => {
    const files = Array.from(e.target.files || []);
    setError("");
    setBatchResult(null);
    setBatchFiles(files);
  };

  const handleRunBatch = async () => {
    // Sign-in gate FIRST, before any validation or any network call. This is a
    // public page: an unauthenticated visitor must be sent to log in, never
    // shown an API failure.
    if (requireLogin({ panel: "batch", mode })) return;

    setError("");
    setBatchResult(null);

    if (batchFiles.length === 0) {
      setError("Select at least one image to compare.");
      setRunState("error");
      return;
    }
    if (mode === "one_to_many" && !referenceFile) {
      setError("Select a reference image to compare every upload against.");
      setRunState("error");
      return;
    }
    // The API records the lawful basis verbatim against EVERY item in the
    // batch, not once for the batch. Not defaulted: the point of the field is
    // that a person had to state a reason.
    if (!lawfulBasis.trim()) {
      setError("State a lawful basis before running this batch.");
      setRunState("error");
      return;
    }

    setRunState("running");
    try {
      setBatchResult(
        await runBatch({
          mode,
          referenceFile,
          probeFiles: batchFiles,
          lawfulBasis: lawfulBasis.trim(),
        }),
      );
      setRunState("complete");
    } catch (err) {
      setError(err.message);
      setRunState("error");
    }
  };

  return (
    <div className="im-batch-panel">
      {/* Mode selector. The three modes answer different questions, so the
          operator picks rather than the UI guessing. */}
      <div className="im-batch-modes" role="radiogroup" aria-label="Batch mode">
        {[
          ["one_to_many", "One reference vs all", "Compare a single suspect against every uploaded image"],
          ["pair", "Independent pairs", "Each upload compared against the enrolled gallery pairwise"],
          ["gallery", "Search gallery", "Search each upload against enrolled subjects"],
        ].map(([id, label, hint]) => (
          <button
            key={id}
            type="button"
            role="radio"
            aria-checked={mode === id}
            className={mode === id ? "is-active" : ""}
            title={hint}
            onClick={() => {
              setMode(id);
              setBatchResult(null);
              setError("");
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {mode === "one_to_many" && (
        <label className="im-batch-drop im-batch-reference">
          <input
            type="file"
            accept="image/*"
            aria-label="Reference image compared against every upload"
            onChange={(e) => {
              setReferenceFile(e.target.files?.[0] || null);
              setBatchResult(null);
              setError("");
            }}
          />
          <span className="im-upload-mark">1</span>
          <strong>Reference image</strong>
          <small>
            {referenceFile
              ? `${referenceFile.name} — compared against every upload below`
              : "The one face every uploaded image is compared against"}
          </small>
        </label>
      )}

      <label className="im-batch-drop">
        <input
          type="file"
          accept="image/*"
          multiple
          aria-label="Upload multiple face images for batch search"
          onChange={handleFilesChange}
        />
        <span className="im-upload-mark">+</span>
        <strong>{mode === "one_to_many" ? "Images to compare" : "Probe images"}</strong>
        <small>
          {batchFiles.length > 0
            ? `${batchFiles.length} images selected${batchFiles.length > 50 ? " — over the 50 limit" : ""}`
            : "Select up to 50 JPG / PNG files"}
        </small>
      </label>

      {/* Recorded verbatim against EVERY item, not once per batch. */}
      <label className="im-lawful-field">
        <span>Lawful basis (required, recorded against every item in the batch)</span>
        <input
          type="text"
          value={lawfulBasis}
          maxLength={500}
          placeholder="e.g. Operation Redwood, warrant ref 2026/114"
          onChange={(e) => setLawfulBasis(e.target.value)}
        />
      </label>

      {batchFiles.length > 0 && (
        <div className="im-batch-file-list">
          <b>Batch Queue ({batchFiles.length} files):</b>
          <ul>
            {batchFiles.slice(0, 8).map((f) => (
              <li key={f.name}>📄 {f.name} ({(f.size / 1024).toFixed(1)} KB)</li>
            ))}
            {batchFiles.length > 8 && <li>... and {batchFiles.length - 8} more files</li>}
          </ul>
        </div>
      )}

      <div className="im-state-panel">
        <div>
          <span>{runState === "running" ? `Processing ${batchFiles.length} images...` : batchResult ? `Complete — ${batchResult.succeeded}/${batchResult.submitted} processed` : "Batch ready"}</span>
          <strong>{runState === "running" ? "..." : batchResult ? `${batchResult.succeeded}/${batchResult.submitted}` : "0"}</strong>
        </div>
        <div className="im-progress">
          <i style={{ width: runState === "running" ? "80%" : batchResult ? "100%" : "0%" }} />
        </div>
      </div>

      {error && (
        <p className="im-error" role="alert">{error}</p>
      )}

      {batchResult && (
        <div className="im-batch-results-table">
          <h4>
            Batch results — {batchResult.succeeded} of {batchResult.submitted} processed
            {batchResult.failed > 0 && `, ${batchResult.failed} failed`}
          </h4>
          <p className="im-batch-threshold">
            Decision threshold {batchResult.threshold}. Similarity is shown to 4
            decimal places so how close a decision fell to the line is visible.
          </p>
          <table>
            <thead>
              <tr>
                <th>Image</th>
                <th>Status</th>
                <th>{batchResult.mode === "gallery" ? "Top candidate" : "Similarity"}</th>
                <th>{batchResult.mode === "gallery" ? "Score" : "Decision"}</th>
                <th>Quality</th>
                <th>Audit hash</th>
              </tr>
            </thead>
            <tbody>
              {batchResult.results.map((res) => {
                const top = res.candidates?.[0];
                return (
                  <tr key={res.index} className={res.status !== "ok" ? "row-error" : ""}>
                    <td><strong>{res.label}</strong></td>
                    <td>
                      <span className={`im-status-pill ${res.status}`}>
                        {res.status === "ok" ? "✓ OK" : "✗ Error"}
                      </span>
                    </td>
                    <td>
                      {res.status !== "ok"
                        ? res.error
                        : batchResult.mode === "gallery"
                          ? (top?.subject_id ?? "no candidate")
                          : res.similarity?.toFixed(4)}
                    </td>
                    <td>
                      {res.status !== "ok"
                        ? "—"
                        : batchResult.mode === "gallery"
                          ? (top ? top.score.toFixed(4) : "—")
                          : res.verified
                            ? "Supports same person"
                            : "Not supported"}
                    </td>
                    <td>
                      {res.probe_quality != null
                        ? `${(res.probe_quality * 100).toFixed(0)}%`
                        : "—"}
                    </td>
                    <td>
                      <code>{res.audit_hash ? `${res.audit_hash.slice(0, 12)}…` : "—"}</code>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="im-batch-notice">{batchResult.notice}</p>
        </div>
      )}

      <button
        type="button"
        className="im-launch"
        onClick={handleRunBatch}
        disabled={runState === "running" || batchFiles.length === 0}
      >
        {runState === "running" ? "Processing Batch Queue..." : batchResult ? "Run Batch Again" : "Launch Batch Processing"}
      </button>
      <p className="im-secure-line">Secure · Encrypted · Live API: {`${imatchApiUrl}/batch`}</p>
    </div>
  );
}

// ─── ComparePanel — real 1:1 verification UI ──────────────────────────────────
function ComparePanel() {
  const requireLogin = useLoginGate();
  const [refFile, setRefFile] = useState(null);
  const [probeFile, setProbeFile] = useState(null);
  const [refPreview, setRefPreview] = useState("");
  const [probePreview, setProbePreview] = useState("");
  const [runState, setRunState] = useState("idle"); // idle | running | complete | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [lawfulBasis, setLawfulBasis] = useState("");

  // Build preview URLs
  useEffect(() => {
    if (!refFile) { setRefPreview(""); return; }
    const url = URL.createObjectURL(refFile);
    setRefPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [refFile]);

  useEffect(() => {
    if (!probeFile) { setProbePreview(""); return; }
    const url = URL.createObjectURL(probeFile);
    setProbePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [probeFile]);

  const handleRun = async () => {
    if (requireLogin({ panel: "compare" })) return;

    setError("");
    setResult(null);

    // The API requires a stated lawful basis and records it verbatim in the
    // audit chain. It is deliberately NOT defaulted to a placeholder here: the
    // point of the field is that a person had to type a reason.
    if (!lawfulBasis.trim()) {
      setError("State a lawful basis for this comparison before running it.");
      setRunState("error");
      return;
    }

    setRunState("running");
    try {
      const n = normalizeVerifyResult(
        await runVerification({
          referenceFile: refFile,
          probeFile,
          lawfulBasis: lawfulBasis.trim(),
        }),
      );

      // Only two decisions are reported, because the API returns exactly two:
      // a similarity and a boolean against one threshold. An "inconclusive"
      // middle band would be a decision rule the engine does not apply.
      setResult({
        ...n,
        score: n.similarity,
        label: n.verified ? "same_person" : "different_person",
        qualityRef: n.reference.quality,
        livenessRef: n.reference.liveness,
        qualityProbe: n.probe.quality,
        livenessProbe: n.probe.liveness,
        reviewRequired: !n.reference.qualityAccepted || !n.probe.qualityAccepted,
      });
      setRunState("complete");
    } catch (err) {
      setError(err.message);
      setRunState("error");
    }
  };

  const labelCfg = result ? (LABEL_CONFIG[result.label] ?? LABEL_CONFIG.unknown) : null;
  // Similarity is a cosine in [-1, 1]; a negative score must not render as a
  // negative percentage bar, and must not be silently clamped away either.
  const scorePct = result ? `${(result.score * 100).toFixed(1)}%` : "—";

  return (
    <div className="im-compare-panel">
      {/* Dual upload zone */}
      <div className="im-compare-uploads">
        <FaceDropZone
          id="compare-ref"
          label="Reference Image"
          preview={refPreview}
          onChange={(f) => { setRefFile(f); setResult(null); setError(""); }}
        />
        <div className="im-compare-vs" aria-hidden="true">VS</div>
        <FaceDropZone
          id="compare-probe"
          label="Probe Image"
          preview={probePreview}
          onChange={(f) => { setProbeFile(f); setResult(null); setError(""); }}
        />
      </div>

      {/* Lawful basis - required by the API, recorded verbatim in the audit chain */}
      <label className="im-lawful-field">
        <span>Lawful basis for this comparison (required, recorded in the audit log)</span>
        <input
          type="text"
          value={lawfulBasis}
          maxLength={500}
          placeholder="e.g. Operation Redwood, warrant ref 2026/114"
          onChange={(event) => setLawfulBasis(event.target.value)}
        />
      </label>

      {/* Score bar */}
      <div className="im-state-panel">
        <div>
          <span>{runState === "running" ? "AI model comparing embeddings" : result ? labelCfg.text : "Comparison ready"}</span>
          <strong>{runState === "running" ? "…" : scorePct}</strong>
        </div>
        <div className="im-progress">
          {/* Cosine ranges [-1,1]; clamp only the BAR width so a negative score
              does not render as a negative element. The numeric score above is
              shown unclamped. */}
          <i style={{ width: runState === "running" ? "60%" : result ? `${Math.max(0, Math.min(100, result.score * 100))}%` : "0%" }} />
        </div>
        <ul>
          {result
            ? [
                `Cosine similarity: ${result.score.toFixed(4)} (threshold ${result.threshold ?? "—"})`,
                `Reference — quality ${(result.qualityRef * 100).toFixed(1)}% · deepfake-artifact risk ${(result.reference.deepfakeRisk * 100).toFixed(1)}%`,
                `Probe — quality ${(result.qualityProbe * 100).toFixed(1)}% · deepfake-artifact risk ${(result.probe.deepfakeRisk * 100).toFixed(1)}%`,
                // Both figures below are HEURISTICS. The backend reports
                // certified:false on every liveness block; that qualifier is
                // rendered here rather than dropped, so nobody reads these as
                // anti-spoofing or as a trained deepfake classifier.
                `Capture heuristics (not certified anti-spoofing): ref ${(result.livenessRef * 100).toFixed(1)}% · probe ${(result.livenessProbe * 100).toFixed(1)}%`,
                result.explanation,
              ].filter(Boolean).map((item) => <li key={item}>{item}</li>)
            : ["Upload reference + probe images", "Runs full ArcFace pipeline", "Returns cosine similarity score"].map((item) => (
                <li key={item}>{item}</li>
              ))}
        </ul>
      </div>

      {/* Result card */}
      {result && (
        <div className="im-compare-result" aria-live="polite" style={{ borderColor: labelCfg.color }}>
          <div className="im-compare-verdict" style={{ background: labelCfg.bg, color: labelCfg.color }}>
            <span className="im-verdict-icon">{labelCfg.icon}</span>
            <span className="im-verdict-text">{labelCfg.text}</span>
            <span className="im-verdict-score">{Math.round(result.score * 100)}% similarity</span>
          </div>
          <div className="im-compare-meta">
            {/* Rendered from the value the backend reports for THIS comparison.
                Never hardcode it: the optimum is model-specific (w600k_r50
                tunes to 0.20, glintr100 to 0.22) and this panel previously
                stated a fixed ">=42% = same person" rule that the engine had
                stopped using -- a decision rule shown to an operator that the
                system was not actually applying. */}
            <div>
              <b>Decision threshold</b>
              <span>
                {typeof result.threshold === "number"
                  ? `similarity ≥ ${result.threshold.toFixed(2)} → supports same person`
                  : "reported by engine per comparison"}
              </span>
            </div>
            {result.reviewRequired && (
              <div className="im-compare-review">⚠ Human review recommended</div>
            )}
            {result.auditHash && (
              <div>
                <b>Audit hash</b>
                <code>{result.auditHash.slice(0, 24)}…</code>
              </div>
            )}
          </div>
        </div>
      )}

      {error && (
        <p className="im-error" role="alert">{error}</p>
      )}

      <button
        type="button"
        className="im-launch"
        onClick={handleRun}
        disabled={runState === "running" || !refFile || !probeFile}
      >
        {runState === "running" ? "Comparing…" : result ? "Run Again" : "Run Comparison"}
      </button>
      <p className="im-secure-line">Secure · Encrypted · Live backend: {`${imatchApiUrl}/verify`}</p>
    </div>
  );
}

// ─── FaceDropZone ─────────────────────────────────────────────────────────────
function FaceDropZone({ id, label, preview, onChange }) {
  return (
    <label className="im-compare-drop" htmlFor={id}>
      <input
        id={id}
        type="file"
        accept="image/*"
        aria-label={label}
        onChange={(e) => onChange(e.target.files?.[0] || null)}
      />
      {preview ? (
        <img className="im-photo-preview" src={preview} alt={label} />
      ) : (
        <>
          <span className="im-upload-mark" aria-hidden="true">+</span>
          <strong>{label}</strong>
          <small>JPG · PNG · WEBP · HEIC</small>
        </>
      )}
    </label>
  );
}

// ─── SingleSearchPanel — original single-image logic ─────────────────────────
function SingleSearchPanel({ step, activeMode }) {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  // Labels describe what the backend actually computes. See CLAIMS.md.
  // - "Image Quality & Capture Check" is security/liveness.py: a passive
  //   single-frame texture/moire/colour heuristic. It is NOT presentation-attack
  //   detection and will not stop a printed photo or a replay attack.
  // - "Synthetic-Media Artifact Screen" is security/deepfake_detector.py: an FFT
  //   smoothness + checkerboard heuristic, not a trained deepfake classifier.
  //   Advisory only.
  // "Auto-enhance" was removed: it had no backend implementation at all, so the
  // checkbox told users their image was being processed when nothing happened.
  const [selectedChecks, setSelectedChecks] = useState({
    "Image Quality & Capture Check": true,
    "Synthetic-Media Artifact Screen": true,
    "Quality Assessment": true,
  });
  // This panel previews the search; it no longer runs one. The button below
  // opens the chooser, and the comparison happens in the product.
  const runState = "idle";
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const options = [
    "Image Quality & Capture Check",
    "Synthetic-Media Artifact Screen",
    "Quality Assessment",
  ];
  const mode = searchModes.find((item) => item.id === activeMode) ?? searchModes[0];
  const scorePercent = result ? `${Math.round(result.matchScore * 100)}%` : step.score;
  const panelLabel = result ? result.decision.replaceAll("_", " ") : step.label;
  const panelResults = result
    ? [
        `Quality ${Math.round(result.quality * 100)}%`,
        `Liveness ${Math.round(result.liveness * 100)}%`,
        result.reviewRequired ? "Human review required" : "Decision ready",
      ]
    : step.results;

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    setError("");
    setResult(null);
    setSelectedFile(file || null);
  };

  const handleCheckChange = (option) => {
    setSelectedChecks((current) => ({
      ...current,
      [option]: !current[option],
    }));
  };

  // "Launch Face Search" is the way into the product, not a demo run: it opens
  // the Individual-vs-Investigator chooser, which is where the real single
  // comparison and the full workspace both start.
  const handleLaunch = () => {
    navigate("/choose-role", { state: { intent: { panel: "single", mode: activeMode } } });
  };

  return (
    <form
      className={`im-single-inner ${result ? "complete" : step.mode}`}
      aria-label="iMatch Single Search"
      onSubmit={(event) => event.preventDefault()}
    >
      <div className="im-console-main">
        <label className="im-drop-zone">
          <input type="file" accept="image/*" aria-label="Upload face image" onChange={handleFileChange} />
          {previewUrl ? (
            <img className="im-photo-preview" src={previewUrl} alt="Selected face search input" />
          ) : (
            <>
              <span className="im-upload-mark" aria-hidden="true">+</span>
              <strong>{mode.title}</strong>
              <small>{mode.formats}</small>
            </>
          )}
        </label>

        <div className="im-preview-stack" aria-hidden={!previewUrl}>
          {previewUrl ? (
            <img src={previewUrl} alt="AI scan preview" />
          ) : (
            <video src={faceSearchVideo} autoPlay muted loop playsInline />
          )}
          <span className="im-preview-scan" />
          <span className="im-landmark-label">468 landmark points</span>
        </div>
      </div>

      {/* URL import removed: the API accepts multipart uploads only, so this
          field was collected and silently discarded. Re-adding it requires a
          real fetch endpoint with SSRF protection (scheme allowlist, DNS
          pinning, RFC1918/loopback/link-local blocking, redirect cap, response
          size limit) -- see CLAIMS.md. */}

      <div className="im-mode-summary" aria-live="polite">
        {mode.summary.map((item) => (
          <span key={item}>{item}</span>
        ))}
      </div>

      <div className="im-option-grid">
        {options.map((option) => (
          <label key={option}>
            <input
              type="checkbox"
              checked={selectedChecks[option]}
              onChange={() => handleCheckChange(option)}
            />
            <span>{option}</span>
          </label>
        ))}
      </div>

      <div className="im-state-panel">
        <div>
          <span>{runState === "running" ? "AI model analyzing" : panelLabel}</span>
          <strong>{runState === "running" ? "..." : scorePercent}</strong>
        </div>
        <div className="im-progress">
          <i style={{ width: runState === "running" ? "72%" : scorePercent }} />
        </div>
        <ul>
          {panelResults.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      {result && (
        <div className="im-ai-results" aria-live="polite">
          <span>AI results</span>
          <div>
            <b>Match</b>
            <strong>{Math.round(result.matchScore * 100)}%</strong>
          </div>
          <div>
            <b>Quality</b>
            <strong>{Math.round(result.quality * 100)}%</strong>
          </div>
          <div>
            <b>Liveness</b>
            <strong>{Math.round(result.liveness * 100)}%</strong>
          </div>
          {result.matches.length > 0 && (
            <ol>
              {result.matches.map((match) => (
                <li key={match.id}>
                  <span>{match.id}</span>
                  <b>{Math.round(match.score * 100)}%</b>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}

      {error && (
        <p className="im-error" role="alert">
          {error}
        </p>
      )}

      <button type="button" className="im-launch" onClick={handleLaunch} disabled={runState === "running"}>
        {runState === "running" ? "Analyzing Photo" : result ? "Run Again" : "Launch Face Search"}
      </button>
      <p className="im-secure-line">Secure - Encrypted - Tenant isolated - AI endpoint: {imatchApiUrl}</p>
    </form>
  );
}
