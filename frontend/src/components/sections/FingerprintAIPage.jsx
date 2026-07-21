import { useEffect, useRef, useState } from "react";
import handImage from "../../assets/hand-hd.png";
import "./FingerprintAIPage.css";

const valueItems = [
  "Latent Print Enhancement",
  "Minutiae Detection",
  "Candidate Ranking",
  "Examiner Review",
  "Report Automation",
];

const problemCards = [
  [
    "Poor Latent Quality",
    "Smudged, partial, or low-contrast prints require heavy manual enhancement before comparison.",
  ],
  [
    "Time-Consuming Comparison",
    "Examiners manually mark ridge endings, bifurcations, ridge flow, and candidate differences.",
  ],
  [
    "Report Preparation Burden",
    "Documentation, annotations, screenshots, notes, and conclusion formatting take extra time.",
  ],
];

const capabilities = [
  ["Fingerprint Enhancement", "Improve ridge visibility in weak, smudged, partial, or noisy fingerprint evidence."],
  ["Ridge Segmentation", "Separate ridge flow from background texture, lift artifacts, and surface noise."],
  ["Minutiae Detection", "Identify ridge endings, bifurcations, cores, deltas, and ridge-flow patterns."],
  ["Quality Assessment", "Classify evidence as high quality, partial, weak, or unsuitable for comparison."],
  ["Candidate Ranking", "Return ranked candidate matches with similarity scores and comparison support."],
  ["Examiner Review", "Enable manual correction, annotation, notes, and expert-controlled conclusions."],
];

const workflowSteps = [
  {
    title: "Upload Evidence",
    body: "User uploads a latent fingerprint image, scanned lift, or case evidence file.",
    score: "18%",
    status: "Evidence intake ready",
  },
  {
    title: "Enhance Print",
    body: "AI improves ridge clarity, contrast, noise, and background separation.",
    score: "42%",
    status: "Ridge enhancement active",
  },
  {
    title: "Extract Features",
    body: "System detects ridge endings, bifurcations, core, delta, and ridge-flow patterns.",
    score: "68%",
    status: "Minutiae map generated",
  },
  {
    title: "Compare Candidates",
    body: "AI compares the print against known or reference fingerprints and ranks possible matches.",
    score: "84%",
    status: "Candidate ranking ready",
  },
  {
    title: "Examiner Review",
    body: "Forensic expert reviews AI markings, edits annotations, verifies details, and adds expert notes.",
    score: "92%",
    status: "Manual review required",
  },
  {
    title: "Generate Report",
    body: "System generates an examiner-ready forensic report with images, notes, match evidence, and audit trail.",
    score: "100%",
    status: "Examiner-ready report",
  },
];

const featureCards = [
  ["Latent Print Enhancement", "Improve ridge visibility in smudged, low-light, partial, or noisy fingerprints."],
  ["Minutiae Detection", "Automatically identify ridge endings, bifurcations, cores, deltas, and ridge-flow patterns."],
  ["Fingerprint Quality Scoring", "Classify evidence as high quality, partial, weak, or unsuitable for comparison."],
  ["Candidate Match Ranking", "Return ranked candidate matches with similarity scores and visual comparison support."],
  ["Examiner Review Workspace", "Side-by-side comparison, overlay alignment, annotations, and expert notes."],
  ["Forensic Report Generator", "Generate structured reports with evidence metadata, quality maps, annotations, and review fields."],
];

const securityCards = [
  ["Encrypted Evidence Storage", "Protect uploaded fingerprint evidence with secure storage workflows."],
  ["Role-Based Access", "Control examiner, admin, reviewer, and analyst permissions."],
  ["Full Audit Trail", "Track uploads, enhancements, edits, reviews, and report exports."],
  ["Chain-of-Custody Logs", "Maintain structured records for sensitive evidence workflows."],
  ["Tamper-Aware Case History", "Preserve important changes and review activity."],
  ["Export Control for Reports", "Control PDF report generation, downloads, and sharing."],
];

const useCases = [
  ["Police Fingerprint Units", "Accelerate latent print preparation, comparison support, and examiner documentation."],
  ["Digital Forensic Labs", "Connect fingerprint findings with broader digital evidence and case workflows."],
  ["Crime Investigation Departments", "Prepare trace evidence for expert review and structured reporting."],
  ["Forensic Science Institutes", "Support repeatable analysis, training, and examiner-led validation."],
  ["Legal Case Documentation Teams", "Produce structured evidence packets with notes, timestamps, and audit history."],
  ["Research and Training Labs", "Study ridge patterns, minutiae quality, and examination workflows with clear outputs."],
];

export function FingerprintAIPage() {
  const [activeStep, setActiveStep] = useState(0);
  const stepRefs = useRef([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActiveStep(Number(visible.target.dataset.step));
      },
      { threshold: [0.42, 0.62, 0.8] }
    );
    stepRefs.current.forEach((node) => node && observer.observe(node));
    return () => observer.disconnect();
  }, []);

  return (
    <>
      <section id="top" className="fp-hero fp-section" aria-labelledby="fingerprint-title">
        <FingerprintBackdrop />
        <div className="fp-copy">
          <p className="fp-badge">AI-Powered Fingerprint Forensics</p>
          <h1 id="fingerprint-title">Automated Fingerprint Examination, From Latent Print to Forensic Report</h1>
          <p>
            NexGen Fingerprint AI enhances poor-quality fingerprints, detects
            minutiae, compares candidate matches, assists examiner review, and
            generates structured forensic reports.
          </p>
          <div className="fp-actions">
            <a href="#briefing">Request Demo</a>
            <a href="#workflow">View Workflow</a>
          </div>
          <ul className="fp-value-row" aria-label="Fingerprint AI value points">
            {valueItems.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <FingerprintAnalysisConsole stage={workflowSteps[0]} />
      </section>

      <section className="fp-section fp-problem" style={{ "--hand-bg": `url(${handImage})` }}>
        <div className="fp-hand-title" aria-label="Automated Fingerprint Examination">
          Automated Fingerprint Examination
        </div>
      </section>

      <section className="fp-section fp-bottlenecks">
        <div className="fp-heading">
          <p className="fp-kicker">Examination Bottlenecks</p>
          <h2>Manual Fingerprint Examination Is Still Slow and Repetitive</h2>
        </div>
        <div className="fp-card-grid three">
          {problemCards.map(([title, body], index) => (
            <article className="fp-card" key={title} style={{ "--delay": `${index * 80}ms` }}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="platform" className="fp-section fp-solution">
        <div className="fp-heading">
          <p className="fp-kicker">AI-Assisted, Examiner-Controlled</p>
          <h2>AI-Assisted Fingerprint Intelligence for Modern Forensic Teams</h2>
          <p>
            NexGen Fingerprint AI helps automate repetitive parts of fingerprint
            examination while keeping examiner review and final conclusions under
            expert control.
          </p>
        </div>
        <div className="fp-card-grid">
          {capabilities.map(([title, body], index) => (
            <article className="fp-card compact" key={title} style={{ "--delay": `${index * 55}ms` }}>
              <FingerprintIcon />
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="workflow" className="fp-section fp-workflow">
        <div className="fp-heading">
          <p className="fp-kicker">Scroll Workflow</p>
          <h2>From Latent Print Upload to Examiner-Ready Report</h2>
        </div>
        <div className="fp-workflow-grid">
          <div className="fp-workflow-visual">
            <FingerprintAnalysisConsole stage={workflowSteps[activeStep]} workflow />
          </div>
          <div className="fp-steps">
            {workflowSteps.map((step, index) => (
              <article
                className={activeStep === index ? "fp-step active" : "fp-step"}
                key={step.title}
                data-step={index}
                ref={(node) => {
                  stepRefs.current[index] = node;
                }}
              >
                <span>Step {String(index + 1).padStart(2, "0")}</span>
                <h3>{step.title}</h3>
                <p>{step.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="fp-section">
        <div className="fp-heading">
          <p className="fp-kicker">Capabilities</p>
          <h2>Built for Real Fingerprint Examination Workflows</h2>
        </div>
        <div className="fp-card-grid">
          {featureCards.map(([title, body], index) => (
            <article className="fp-card compact" key={title} style={{ "--delay": `${index * 55}ms` }}>
              <FingerprintIcon />
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="fp-section fp-preview" aria-labelledby="dashboard-title">
        <div className="fp-heading">
          <p className="fp-kicker">Forensic Workstation</p>
          <h2 id="dashboard-title">Interactive Dashboard Preview</h2>
        </div>
        <div className="fp-workstation">
          <div className="fp-panel">
            <strong>Uploaded Latent Fingerprint</strong>
            <FingerprintPlate />
          </div>
          <div className="fp-panel featured">
            <strong>Enhanced Fingerprint + Minutiae</strong>
            <FingerprintPlate enhanced />
          </div>
          <div className="fp-panel">
            <strong>Candidate Matches</strong>
            {["#1 92.4% similarity", "#2 88.1% similarity", "#3 81.6% similarity", "#4 77.2% similarity", "#5 73.9% similarity"].map((row) => (
              <span className="fp-candidate" key={row}>{row}</span>
            ))}
          </div>
          <div className="fp-observations">
            <span>Ridge Ending: 42 detected</span>
            <span>Bifurcation: 31 detected</span>
            <span>Quality Score: 84/100</span>
            <span>Manual Review Required</span>
            <span>Examiner Notes Pending</span>
            <button type="button">Generate Forensic Report</button>
          </div>
        </div>
      </section>

      <section id="research" className="fp-section fp-report">
        <div className="fp-heading">
          <p className="fp-kicker">Report Automation</p>
          <h2>Court-Ready Structure. Examiner-Controlled Conclusion.</h2>
          <p>
            NexGen Fingerprint AI automatically prepares the forensic report
            structure while keeping final forensic conclusions under examiner
            control.
          </p>
        </div>
        <article className="fp-report-card">
          {["Case ID", "Evidence ID", "Upload timestamp", "Image quality score", "Enhancement log", "Minutiae map", "Candidate match table", "Annotated comparison images", "Examiner notes", "Verification status", "Chain-of-custody log"].map((item) => (
            <span key={item}>{item}</span>
          ))}
        </article>
      </section>

      <section id="validation" className="fp-section">
        <div className="fp-heading">
          <p className="fp-kicker">Evidence Security</p>
          <h2>Secure Evidence Handling</h2>
        </div>
        <div className="fp-card-grid">
          {securityCards.map(([title, body], index) => (
            <article className="fp-card compact" key={title} style={{ "--delay": `${index * 55}ms` }}>
              <FingerprintIcon />
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="fp-section fp-use">
        <div className="fp-heading">
          <p className="fp-kicker">Use Cases</p>
          <h2>Designed for Forensic Teams</h2>
        </div>
        <div className="fp-card-grid">
          {useCases.map(([title, body], index) => (
            <article className="fp-card compact" key={title} style={{ "--delay": `${index * 55}ms` }}>
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="briefing" className="fp-final fp-section">
        <FingerprintBackdrop />
        <p className="fp-kicker">Deploy Fingerprint AI</p>
        <h2>Bring AI Speed to Fingerprint Forensic Examination</h2>
        <p>
          Enhance, analyze, compare, review, and report fingerprints from one
          forensic-grade platform.
        </p>
        <div className="fp-actions">
          <a href="mailto:access@nexgenforensics.ai">Request Demo</a>
          <a href="mailto:access@nexgenforensics.ai">Contact NexGen Forensics</a>
        </div>
      </section>
    </>
  );
}

function FingerprintDashboard({ stage, workflow = false }) {
  return (
    <aside className={workflow ? "fp-dashboard workflow" : "fp-dashboard"} aria-label="Fingerprint AI examiner workstation">
      <div className="fp-dashboard-top">
        <span>Evidence ID: NG-FP-0429</span>
        <span>Latent Print Examination</span>
      </div>
      <div className="fp-dashboard-main">
        <FingerprintPlate enhanced={workflow} />
        <div className="fp-dashboard-side">
          <strong>{stage.status}</strong>
          <div className="fp-score">
            <span>Quality Score</span>
            <em>{stage.score}</em>
            <i style={{ width: stage.score }} />
          </div>
          {["Candidate #1 - 92.4%", "Manual Review Required", "Report Status: Examiner-ready"].map((row) => (
            <p key={row}>{row}</p>
          ))}
        </div>
      </div>
      <div className="fp-dashboard-bottom">
        <span>Review status: Manual Review Required</span>
        <span>Audit Trail Active</span>
      </div>
    </aside>
  );
}

function FingerprintAnalysisConsole({ stage, workflow = false }) {
  const [activeMode, setActiveMode] = useState("single");
  const modes = [
    ["single", "Single Print", "Drop latent fingerprint, scanned lift, or fingerprint image here"],
    ["compare", "1:1 Compare", "Upload latent and reference fingerprints for comparison"],
    ["batch", "Batch Upload", "Drop a case folder, PDF set, or fingerprint batch manifest"],
    ["url", "URL Import", "Import fingerprint evidence from a secure evidence path"],
  ];
  const activeModeData = modes.find(([id]) => id === activeMode) ?? modes[0];
  const options = [
    "Latent Print Enhancement",
    "Minutiae Detection",
    "Ridge Flow Mapping",
    "Quality Assessment",
    "Candidate Matching",
    "Report Drafting",
  ];
  const metrics = [
    "Quality Score: 84/100",
    "Ridge Ending: 42 detected",
    "Bifurcation: 31 detected",
    "Candidate Rank #1: 92.4% similarity",
    "Review Status: Manual Review Required",
  ];

  return (
    <form className={workflow ? "fp-analysis-console workflow" : "fp-analysis-console"} aria-label="Fingerprint Analysis Console">
      <div className="fp-analysis-head">
        <div>
          <span>NexGen Forensics - Fingerprint AI</span>
          <h3>Fingerprint Analysis Console</h3>
        </div>
        <strong>NG-FP-0429</strong>
      </div>

      <div className="fp-analysis-tabs" role="tablist" aria-label="Fingerprint analysis mode">
        {modes.map(([id, label]) => (
          <button
            type="button"
            role="tab"
            aria-selected={activeMode === id}
            className={activeMode === id ? "active" : ""}
            key={id}
            onClick={() => setActiveMode(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="fp-analysis-main">
        <label className="fp-fingerprint-drop">
          <input type="file" accept="image/*,.pdf" aria-label="Upload latent fingerprint evidence" />
          <span className="fp-upload-mark" aria-hidden="true">+</span>
          <strong>{activeModeData[2]}</strong>
          <small>JPG - JPEG - PNG - WEBP - TIFF - BMP - PDF</small>
        </label>
        <div className="fp-mini-print" aria-hidden="true">
          <FingerprintPlate enhanced={workflow} />
        </div>
      </div>

      <label className="fp-evidence-url">
        <span>Evidence URL</span>
        <input type="url" placeholder="Enter fingerprint image URL or secure evidence path" />
      </label>

      <div className="fp-analysis-options">
        {options.map((option) => (
          <label key={option}>
            <input type="checkbox" defaultChecked />
            <span>{option}</span>
          </label>
        ))}
      </div>

      <div className="fp-analysis-state">
        <div>
          <span>{stage.title}</span>
          <strong>{stage.score}</strong>
        </div>
        <i><b style={{ width: stage.score }} /></i>
        <p>{stage.status}</p>
      </div>

      <div className="fp-result-preview">
        {metrics.map((metric) => (
          <span key={metric}>{metric}</span>
        ))}
      </div>

      <button type="button" className="fp-analysis-launch">
        Launch Fingerprint Analysis
      </button>
      <p className="fp-analysis-secure">Secure - Encrypted - Examiner-Controlled - Audit-Ready</p>
    </form>
  );
}

function FingerprintPlate({ enhanced = false }) {
  return (
    <div className={enhanced ? "fp-print enhanced" : "fp-print"} aria-hidden="true">
      {Array.from({ length: enhanced ? 18 : 10 }).map((_, index) => (
        <i key={index} />
      ))}
    </div>
  );
}

function FingerprintIcon() {
  return (
    <span className="fp-icon" aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}

function FingerprintBackdrop() {
  return (
    <div className="fp-backdrop" aria-hidden="true">
      <span />
      <span />
    </div>
  );
}

export default FingerprintAIPage;
