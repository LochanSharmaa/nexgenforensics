import { useEffect, useMemo, useRef, useState } from "react";
import "./FaceSearchExperience.css";
import faceSearchVideo from "../../assets/facesearch.mp4";
import { imatchApiUrl, runImatchSearch } from "../../services/imatchApi";

const trustItems = [
  "Commercial Face Matching",
  "Liveness Detection",
  "Deepfake Check",
  "Tenant Isolation",
  "99.99% Benchmark Target",
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
    body: "Liveness, deepfake, consent, and quality checks are preserved with recognition scoring for enterprise review workflows.",
    mode: "complete",
    score: "96.8%",
    results: ["Liveness passed", "Deepfake check passed", "Audit entry ready"],
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
      "POST /api/imatch/search",
      "{",
      '  "image": "face-image.jpg",',
      '  "checks": ["liveness", "deepfake", "quality"],',
      '  "threshold": 0.84',
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
  {
    id: "url",
    label: "URL Import",
    title: "Import a remote face image from a secure URL",
    formats: "HTTPS - S3 - Azure Blob - GCS",
    urlLabel: "Image URL",
    placeholder: "Paste secure image URL",
    summary: ["Remote intake", "Source validation", "Cloud path verified"],
  },
];

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
          <p className="im-badge">Enterprise biometric engine - validation target 99.99%</p>
          <h1 id="imatch-title">NexGen iMatch</h1>
          <h2>Enterprise Facial Recognition System</h2>
          <p>
            Advanced facial recognition for commercial face search, identity
            verification, fraud prevention, access control, liveness detection,
            deepfake checks, and secure recognition workflows.
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

function ImatchUploadConsole({ step, hero = false }) {
  const [activeMode, setActiveMode] = useState(searchModes[0].id);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [selectedChecks, setSelectedChecks] = useState({
    "Liveness Check": true,
    "Deepfake Check": true,
    "Quality Assessment": true,
    "Auto-enhance": true,
  });
  const [runState, setRunState] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const options = ["Liveness Check", "Deepfake Check", "Quality Assessment", "Auto-enhance"];
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

  const handleLaunch = async () => {
    setError("");
    setResult(null);
    setRunState("running");

    try {
      const response = await runImatchSearch({
        file: selectedFile,
        mode: activeMode,
        sourceUrl,
        checks: Object.entries(selectedChecks)
          .filter(([, enabled]) => enabled)
          .map(([label]) => label),
      });
      setResult(response);
      setRunState("complete");
    } catch (launchError) {
      setError(launchError.message);
      setRunState("error");
    }
  };

  return (
    <form
      className={`im-upload-console ${result ? "complete" : step.mode}${hero ? " hero" : ""}`}
      aria-label="iMatch Face Search"
      onSubmit={(event) => event.preventDefault()}
    >
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

      <label className="im-url-field">
        <span>{mode.urlLabel}</span>
        <input
          type="url"
          placeholder={mode.placeholder}
          value={sourceUrl}
          onChange={(event) => setSourceUrl(event.target.value)}
        />
      </label>

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

export default FaceSearchExperience;
