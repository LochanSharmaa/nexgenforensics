import "./ProductShowcasePages.css";

/**
 * Data-driven product landing pages for the six showcase products from the
 * home-page deck. One layout, six configs — each page carries its product's
 * accent colour and a mock console echoing its slide screenshot.
 */

function ProductShowcase({ config: c }) {
  return (
    <div
      className="px-page"
      style={{ "--px-accent": c.accent, "--px-soft": c.soft }}
    >
      <section className="px-hero">
        <p className="px-badge">
          <i aria-hidden="true" /> NexGen {c.engine} · {c.badge}
        </p>
        <h1>
          {c.titleA} <span>{c.titleB}</span>
        </h1>
        <p className="px-lead">{c.lead}</p>

        <div className="px-chips">
          {c.chips.map((chip) => (
            <div className="px-chip" key={chip.title}>
              <strong>{chip.title}</strong>
              <span>{chip.value}</span>
            </div>
          ))}
        </div>

        <div className="px-actions">
          <a className="px-cta-filled" href={c.demoHref}>
            {c.ctaLabel ?? "Try Online Demo"}
          </a>
          <a className="px-cta-outline" href={c.secondaryHref ?? "/contact"}>
            {c.secondaryLabel ?? "Request Access"}
          </a>
        </div>
      </section>

      <section className="px-console-wrap" aria-label={`${c.engine} console preview`}>
        <div className="px-console">
          <div className="px-console-top">
            <span className="px-dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            <span className="px-console-name">{c.console.title}</span>
            <em>{c.console.status}</em>
          </div>

          <div className="px-console-body">
            <div className="px-drop-zone">
              <strong>{c.console.drop.title}</strong>
              <span>{c.console.drop.hint}</span>
              <b>{c.console.drop.button}</b>
            </div>

            <div className="px-results">
              <div className="px-verdict" data-tone={c.console.verdict.tone}>
                <strong>{c.console.verdict.label}</strong>
                <span>{c.console.verdict.sub}</span>
              </div>

              <div className="px-score">
                <span>{c.console.score.label}</span>
                <strong>{c.console.score.value}</strong>
              </div>

              {c.console.bars.length > 0 && (
                <div className="px-bars">
                  {c.console.bars.map((bar) => (
                    <div className="px-bar" key={bar.label}>
                      <span>{bar.label}</span>
                      <div className="px-bar-track">
                        <div className="px-bar-fill" style={{ width: `${bar.pct}%` }} />
                      </div>
                      <em>{bar.pct}%</em>
                    </div>
                  ))}
                </div>
              )}

              <div className="px-facts">
                {c.console.facts.map((fact) => (
                  <div key={fact.label}>
                    <span>{fact.label}</span>
                    <strong>{fact.value}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="px-timeline" aria-label="Event timeline">
            {c.console.timeline.map((entry) => (
              <span key={entry}>{entry}</span>
            ))}
          </div>
        </div>
      </section>

      <section className="px-caps">
        <h2>{c.capsTitle}</h2>
        <div className="px-caps-grid">
          {c.capabilities.map((cap) => (
            <article className="px-cap" key={cap.title}>
              <strong>{cap.title}</strong>
              <p>{cap.desc}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="px-final">
        <h2>{c.finalTitle}</h2>
        <p>{c.finalLead}</p>
        <div className="px-actions">
          <a className="px-cta-filled" href={c.demoHref}>
            {c.ctaLabel ?? "Try Online Demo"}
          </a>
          <a className="px-cta-outline" href="/products">
            Explore All Products
          </a>
        </div>
      </section>
    </div>
  );
}

const osint = {
  engine: "oTRACE",
  badge: "AI-Powered OSINT",
  accent: "#6d3bd8",
  soft: "rgba(109, 59, 216, 0.09)",
  titleA: "OSINT Investigation.",
  titleB: "Actionable Public Intelligence.",
  lead: "AI-powered OSINT platform for public intelligence, identity correlation, threat mapping, link analysis, and evidence-ready reporting. Built for investigators, analysts, and agencies.",
  chips: [
    { title: "Identity Correlation", value: "Cross-source matching" },
    { title: "Threat Mapping", value: "Geo-intelligence" },
    { title: "Link Analysis", value: "Entity relationships" },
    { title: "Evidence Reporting", value: "Court-ready export" },
  ],
  demoHref: "/demo/osint",
  console: {
    title: "oTRACE · Entity Search",
    status: "Investigation Ready",
    drop: {
      title: "Search entities",
      hint: "Names, aliases, emails, usernames, phones, domains, or public records",
      button: "Search",
    },
    verdict: { label: "Primary entity resolved", sub: "John A. Doe · New York, USA", tone: "ok" },
    score: { label: "Risk & Confidence", value: "97.6%" },
    bars: [
      { label: "Suspicious affiliations", pct: 86 },
      { label: "Adverse media", pct: 54 },
      { label: "Sanctions watchlist", pct: 22 },
      { label: "Dark search exposure", pct: 18 },
    ],
    facts: [
      { label: "Linked entities", value: "128" },
      { label: "Identified", value: "247" },
      { label: "Web pages", value: "312" },
      { label: "Social posts", value: "1,254" },
    ],
    timeline: [
      "09:12 · Profile discovered",
      "09:36 · Aliases identified",
      "10:02 · Locations mapped",
      "10:24 · Links analyzed",
      "11:04 · Report generated",
    ],
  },
  capsTitle: "From data to decisions. Backed by intelligence.",
  capabilities: [
    { title: "Identity Correlation", desc: "Cross-source matching to unify identities across emails, usernames, phones, documents, and public records." },
    { title: "Threat Mapping", desc: "Map threats, events, and infrastructure across geographies and visualize connections in real time." },
    { title: "Public Intelligence Discovery", desc: "AI-powered discovery across open sources, social media, web, dark web, leaks, and public databases." },
    { title: "Evidence-Ready Reporting", desc: "Generate court-admissible reports with evidence logs, source citations, and chain of custody." },
  ],
  finalTitle: "Turn public data into verified intelligence.",
  finalLead: "NexGen oTRACE empowers teams to move from scattered open sources to confident, documented findings.",
};

const deepfake = {
  engine: "DeepVision",
  badge: "AI-Powered Verification",
  accent: "#8a6116",
  soft: "rgba(138, 97, 22, 0.1)",
  titleA: "Deepfake Detection.",
  titleB: "Trust What You See.",
  lead: "Advanced AI models analyze videos, images, and audio to detect deepfakes with industry-leading accuracy. Verify authenticity. Protect truth.",
  chips: [
    { title: "Multi-Modal", value: "Video · Image · Audio" },
    { title: "High Accuracy", value: "AI models" },
    { title: "Fast & Scalable", value: "12s avg analysis" },
    { title: "Privacy", value: "Focused by design" },
  ],
  demoHref: "/demo/deepfake-detection",
  console: {
    title: "DeepVision · Video Analysis",
    status: "Analysis Complete",
    drop: {
      title: "Upload video file",
      hint: "Supports MP4, MOV, AVI, MKV · Max file size 2GB",
      button: "Choose File",
    },
    verdict: { label: "Deepfake detected", sub: "High probability of manipulation", tone: "alert" },
    score: { label: "Confidence Score", value: "92.7%" },
    bars: [
      { label: "Face", pct: 94 },
      { label: "Eye movement", pct: 91 },
      { label: "Facial artifacts", pct: 89 },
      { label: "Audio sync", pct: 72 },
    ],
    facts: [
      { label: "Analyzed in", value: "12.4s" },
      { label: "Model", value: "DeepVision v3.2" },
      { label: "Modality", value: "Video" },
      { label: "Frames checked", value: "4,318" },
    ],
    timeline: [
      "Frame-by-frame analysis",
      "Audio-visual consistency",
      "AI explainability",
      "Forensic report export",
    ],
  },
  capsTitle: "Built for accuracy. Designed for impact.",
  capabilities: [
    { title: "AI Video Analysis", desc: "Detects facial swaps, reenactments, and synthetic video generation frame by frame." },
    { title: "Image Forensics", desc: "Identifies AI-generated images and manipulated visual content." },
    { title: "Audio Verification", desc: "Analyzes voice cloning and audio manipulation at scale." },
    { title: "API & Integrations", desc: "Seamless API access for your platforms and workflows." },
  ],
  finalTitle: "Fighting misinformation. Protecting truth.",
  finalLead: "NexGen DeepVision helps you verify authenticity in a world of AI-generated deception.",
};

const crime3d = {
  engine: "SceneRebuild 3D",
  badge: "Spatial Reconstruction",
  accent: "#8c2137",
  soft: "rgba(140, 33, 55, 0.09)",
  titleA: "Forensic 3D",
  titleB: "Crime Scene Reconstruction.",
  ctaLabel: "Book a Live Demo",
  secondaryLabel: "Explore All Products",
  secondaryHref: "/products",
  lead: "Recreate crime scenes in 3D with object placement, movement paths, and event animation. SceneRebuild 3D turns photos, scans, floor plans, CCTV, and evidence markers into one spatial investigation model.",
  chips: [
    { title: "3D Scene Modeling", value: "Spatially accurate" },
    { title: "Object Placement", value: "Evidence positioning" },
    { title: "Movement Paths", value: "Subject & object motion" },
    { title: "Event Animation", value: "Replay chronology" },
  ],
  demoHref: "/contact",
  console: {
    title: "SceneRebuild · Scene Build",
    status: "Reconstruction Ready",
    drop: {
      title: "Import & build scene",
      hint: "Supports JPG, PNG, TIFF, PDF, DWG, LAS, E57, MP4",
      button: "Launch Analysis",
    },
    verdict: { label: "Chain of custody verified", sub: "Case CS-2024-0417-3D-001", tone: "ok" },
    score: { label: "Reconstruction Confidence", value: "97.4%" },
    bars: [
      { label: "Floor plan alignment", pct: 98 },
      { label: "LiDAR registration", pct: 95 },
      { label: "CCTV synchronization", pct: 91 },
      { label: "Evidence coverage", pct: 88 },
    ],
    facts: [
      { label: "Event stages", value: "6" },
      { label: "Evidence objects", value: "24" },
      { label: "Quality score", value: "4.6/5" },
      { label: "Collection date", value: "Apr 17, 2024" },
    ],
    timeline: [
      "14:01 · Entry through door",
      "14:02 · Object moved",
      "14:03 · Incident point",
      "14:05 · Scene interaction",
      "14:07 · Exit",
    ],
  },
  capsTitle: "One spatial model for the whole investigation.",
  capabilities: [
    { title: "Scene Build", desc: "Upload or import photos, scans, floor plans, and CCTV to build your 3D scene." },
    { title: "Object Layout", desc: "Place victims, weapons, vehicles, and evidence markers with spatial accuracy." },
    { title: "Path Mapping", desc: "Track subject and object motion through the reconstructed scene." },
    { title: "Event Replay", desc: "Animate incident chronology with synchronized timestamps and device metadata." },
  ],
  finalTitle: "Reconstruct what happened. Show it in space.",
  finalLead: "SceneRebuild 3D turns fragmented captures into one court-ready spatial investigation model.",
};

const graph = {
  engine: "EvidenceGraph AI",
  badge: "Investigation Graph",
  accent: "#c22a44",
  soft: "rgba(194, 42, 68, 0.08)",
  titleA: "Forensic Evidence",
  titleB: "Graph.",
  lead: "Connects every evidence item — suspect, victim, face, phone, vehicle, location, CCTV, document, fingerprint — into one visual investigation graph with AI-powered entity resolution.",
  chips: [
    { title: "Entity Resolution", value: "AI identity merging" },
    { title: "Relationship Mapping", value: "Visual link discovery" },
    { title: "Cross-Source Correlation", value: "Structured + unstructured" },
    { title: "Evidence Reporting", value: "Provenance & audit trails" },
  ],
  demoHref: "/demo/evidence-graph",
  console: {
    title: "EvidenceGraph · Graph Build",
    status: "High Confidence",
    drop: {
      title: "Drag & drop evidence here",
      hint: "Face · Phone · CCTV · Vehicle · Document · Fingerprint · Location",
      button: "Add to Graph",
    },
    verdict: { label: "Suspect node resolved", sub: "Case 2024_0417_Downtown", tone: "ok" },
    score: { label: "Graph Confidence", value: "97%" },
    bars: [
      { label: "Face match", pct: 95 },
      { label: "Phone activity link", pct: 88 },
      { label: "Location correlation", pct: 82 },
      { label: "Document mentions", pct: 74 },
    ],
    facts: [
      { label: "Linked entities", value: "243" },
      { label: "Evidence sources", value: "12" },
      { label: "Total items", value: "1,248" },
      { label: "Added today", value: "+18" },
    ],
    timeline: [
      "CCTV capture",
      "Face match · 95%",
      "Phone activity",
      "Location ping",
      "Fingerprint match · 98%",
    ],
  },
  capsTitle: "Disparate data. One actionable picture.",
  capabilities: [
    { title: "Entity Resolution", desc: "AI matches and merges identities across sources with high accuracy." },
    { title: "Relationship Mapping", desc: "Discovers and visualizes links between people, objects, events, and places." },
    { title: "Cross-Source Correlation", desc: "Correlates structured and unstructured data across multiple evidence sources." },
    { title: "Evidence-Ready Reporting", desc: "Generates court-admissible reports with provenance and audit trails." },
  ],
  finalTitle: "Every item connected. Every link explained.",
  finalLead: "EvidenceGraph AI unifies your case into one visual graph investigators can interrogate and trust.",
};

const video = {
  engine: "vMATCH",
  badge: "Video Evidence AI",
  accent: "#a84010",
  soft: "rgba(168, 64, 16, 0.09)",
  titleA: "Forensic Video",
  titleB: "Evidence Analysis.",
  ctaLabel: "Book a Live Demo",
  secondaryLabel: "Explore All Products",
  secondaryHref: "/products",
  lead: "Upload a video and let NexGen vMATCH perform advanced forensic analysis across complex footage — detecting persons, vehicles, objects, and critical events with actionable intelligence.",
  chips: [
    { title: "Forensic-Grade Accuracy", value: "99.2%" },
    { title: "Chain of Custody", value: "End-to-end" },
    { title: "Multi-Entity Detection", value: "AI-powered" },
    { title: "Evidence Reporting", value: "Court-admissible" },
  ],
  demoHref: "/contact",
  console: {
    title: "vMATCH · Video Search",
    status: "Analysis Complete",
    drop: {
      title: "Drag & drop video evidence",
      hint: "Supports MP4, MOV, AVI, MKV, WEBM",
      button: "Launch Analysis",
    },
    verdict: { label: "Chain of custody verified", sub: "Clip VID-59328471 · Rank 1 of 18", tone: "ok" },
    score: { label: "Match Confidence", value: "97.9%" },
    bars: [
      { label: "Person detection", pct: 96 },
      { label: "Vehicle detection", pct: 93 },
      { label: "Object recognition", pct: 88 },
      { label: "Event chronology", pct: 85 },
    ],
    facts: [
      { label: "Detected entities", value: "23" },
      { label: "Primary event", value: "Vehicle arrival" },
      { label: "Quality score", value: "93/100" },
      { label: "Collection date", value: "May 14, 2025" },
    ],
    timeline: [
      "09:42:14 · Person enters frame",
      "09:42:21 · Vehicle detected",
      "09:42:39 · Object exchange flagged",
      "09:43:02 · Vehicle exits scene",
    ],
  },
  capsTitle: "Complex footage. Coherent chronology.",
  capabilities: [
    { title: "Person Detection", desc: "Identify individuals across frames with tracked continuity." },
    { title: "Vehicle Detection", desc: "Track vehicles through the scene with entry and exit events." },
    { title: "Object Recognition", desc: "Flag relevant objects and interactions for investigative review." },
    { title: "Event Timeline", desc: "Build incident chronology with timestamps suitable for court review." },
  ],
  finalTitle: "See every event. Prove every second.",
  finalLead: "vMATCH turns hours of complex footage into a documented, reviewable sequence of events.",
};

const caseIntel = {
  engine: "CaseGPT",
  badge: "Case Intelligence AI",
  accent: "#2a64bb",
  soft: "rgba(42, 100, 187, 0.08)",
  titleA: "Forensic Case",
  titleB: "Intelligence.",
  ctaLabel: "Book a Live Demo",
  secondaryLabel: "Explore All Products",
  secondaryHref: "/products",
  lead: "Automate case report drafting, evidence summarization, and timeline generation using AI built for investigative workflows. Save time, reduce manual effort, and deliver court-ready intelligence.",
  chips: [
    { title: "Auto Report Drafting", value: "Structured first drafts" },
    { title: "Evidence Summarization", value: "Cross-source synthesis" },
    { title: "Timeline Generation", value: "Built automatically" },
    { title: "Court-Ready Reporting", value: "Reviewable & exportable" },
  ],
  demoHref: "/contact",
  console: {
    title: "CaseGPT · Case Draft",
    status: "Draft Ready",
    drop: {
      title: "Drop case files, notes, and transcripts",
      hint: "Supports PDF, DOCX, TXT, CSV, JPG, PNG, MP4",
      button: "Generate Report",
    },
    verdict: { label: "Chain of custody verified", sub: "CASE-59328471 · Priority high", tone: "ok" },
    score: { label: "Draft Completeness", value: "96.8%" },
    bars: [
      { label: "Executive summary", pct: 100 },
      { label: "Incident overview", pct: 98 },
      { label: "Evidence summary", pct: 95 },
      { label: "Key findings", pct: 92 },
    ],
    facts: [
      { label: "Timeline events", value: "18" },
      { label: "Report type", value: "Investigation summary" },
      { label: "Quality score", value: "94/100" },
      { label: "Pages drafted", value: "12" },
    ],
    timeline: [
      "09:42 · Device seized",
      "10:05 · CCTV reviewed",
      "10:28 · Witness statement logged",
      "11:16 · Evidence linked",
      "11:48 · Draft generated",
    ],
  },
  capsTitle: "Manual effort out. Court-ready intelligence in.",
  capabilities: [
    { title: "Report Drafting", desc: "Generate structured investigative reports from raw case material." },
    { title: "Evidence Summary", desc: "Condense multi-source evidence into coherent, cited summaries." },
    { title: "Timeline Extraction", desc: "Build incident chronology automatically from documents and media." },
    { title: "Key Findings", desc: "Surface critical points with context for human approval workflows." },
  ],
  finalTitle: "From case files to confident decisions.",
  finalLead: "CaseGPT drafts, summarizes, and structures — investigators review, decide, and sign.",
};

export const OsintProductPage = () => <ProductShowcase config={osint} />;
export const DeepfakeProductPage = () => <ProductShowcase config={deepfake} />;
export const CrimeScene3DProductPage = () => <ProductShowcase config={crime3d} />;
export const EvidenceGraphProductPage = () => <ProductShowcase config={graph} />;
export const VideoAnalysisProductPage = () => <ProductShowcase config={video} />;
export const CaseIntelligenceProductPage = () => <ProductShowcase config={caseIntel} />;
