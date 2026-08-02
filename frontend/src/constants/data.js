export const divisions = [
  {
    id: "osint",
    href: "/products/osint",
    eyebrow: "01 / Tenant Search",
    title: "Search stays inside permitted scope.",
    thesis: "One-to-many face search is isolated by workspace, role, policy, and explicit permissions.",
    products: ["Workspace Index", "Batch Search", "Permission Matrix"],
    proof: ["Tenant", "Role", "Purpose", "Search log"],
  },
  {
    id: "audio",
    href: "/products/deepfake-detection",
    eyebrow: "02 / Authenticity Controls",
    title: "Spoof risk becomes visible.",
    thesis: "Liveness, deepfake, and capture-quality checks help teams trust when a biometric result is reviewable.",
    products: ["Liveness", "Deepfake Check", "Spoof Signals"],
    proof: ["Motion", "Texture", "Artifacts", "Risk score"],
  },
  {
    id: "document",
    href: "/products/3d-crime-scene",
    eyebrow: "03 / Model Governance",
    title: "Improvement is controlled.",
    thesis: "Uploaded data can queue validated fine-tuning jobs, but deployment happens only after benchmark and rollback checks.",
    products: ["Fine-Tune Queue", "Benchmark Gate", "Rollback"],
    proof: ["Train", "Validate", "A/B gate", "Version log"],
  },
  {
    id: "identity",
    href: "/products/imatch",
    eyebrow: "04 / Enterprise Identity",
    title: "Verify the right person.",
    thesis: "High-assurance face verification and biometric search for authorized enterprise workflows.",
    products: ["iMatch", "Face Verification", "Auto-Find"],
    proof: ["Consent check", "Quality scoring", "Encrypted template", "Operator review"],
  },
];

export const evidenceItems = [
  "Consent Record",
  "Face Search Request",
  "Tenant Workspace",
  "Encrypted Template",
  "Liveness Signal",
  "Deepfake Risk Flag",
  "Quality Score",
  "Benchmark Target",
  "Access-Control Policy",
  "Operator ID",
  "Retention Rule",
  "Audit Log Entry",
  "Rollback Version",
  "SLA Metric",
  "Compliance Report",
  "Review Decision",
];

export const research = [
  { title: "Biometric Accuracy", desc: "Benchmark design, error rates, demographic performance review, and documented validation targets." },
  { title: "Privacy Engineering", desc: "Consent-aware intake, data minimization, template encryption, and retention controls." },
  { title: "Tenant Isolation", desc: "Workspace partitioning, role boundaries, and explicit permission controls for commercial clients." },
  { title: "Model Governance", desc: "Lifecycle management, version control, rollback plans, and commercial readiness gates." },
  { title: "Human Review", desc: "Operator decision support, escalation states, and review-required confidence boundaries." },
  { title: "API Reliability", desc: "Rate limiting, latency monitoring, health checks, and resilient hybrid deployments." },
  { title: "Security Controls", desc: "Secure authentication, audit logs, liveness detection, anti-spoofing, and payload protection." },
  { title: "Compliance Reporting", desc: "Exportable reports designed to support regulated enterprise review without making unverified legal claims." },
];

export const correlationSources = [
  { label: "Device Data", meta: "Phones, laptops, storage media" },
  { label: "CCTV", meta: "Video frames, timestamps, locations" },
  { label: "OSINT", meta: "Social profiles, public records" },
  { label: "Voice Sample", meta: "Audio recordings, voiceprints" },
];
