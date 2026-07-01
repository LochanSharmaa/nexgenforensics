export const productLinks = [
  {
    title: "NexGen iMatch",
    category: "AI Facial Recognition System",
    href: "/products/imatch",
    description:
      "Face search, face matching, liveness checks, deepfake detection, and secure biometric recognition workflows.",
  },
  {
    title: "NexGen Fingerprint AI",
    category: "AI Fingerprint Examination Platform",
    href: "/products/fingerprint-ai",
    description:
      "Latent print enhancement, minutiae detection, candidate comparison, examiner review, and forensic report automation.",
  },
  {
    title: "Deepfake Detection",
    category: "Synthetic Media Analysis",
    href: "/products/deepfake-detection",
    description:
      "Detect manipulated facial media, synthetic images, AI-generated videos, and authenticity risk signals.",
  },
  {
    title: "OSINT Intelligence",
    category: "Open-Source Intelligence Platform",
    href: "/products/osint",
    description:
      "Collect, organize, and analyze public-source intelligence for investigation and verification workflows.",
  },
  {
    title: "Evidence Graph",
    category: "Investigation Link Analysis",
    href: "/products/evidence-graph",
    description:
      "Visualize relationships between people, evidence, devices, locations, cases, and digital traces.",
  },
  {
    title: "Case Intelligence",
    category: "AI Case Management",
    href: "/products/case-intelligence",
    description:
      "Summarize case materials, structure investigation notes, manage evidence context, and support analyst workflows.",
  },
  {
    title: "API Platform",
    category: "Forensic AI API",
    href: "/products/api",
    description:
      "Integrate NexGen recognition, analysis, and intelligence APIs into secure internal systems.",
  },
];

export const solutionLinks = [
  {
    title: "Forensic Labs",
    href: "/solutions/forensic-labs",
    description: "AI-assisted analysis workflows for fingerprint, face, media, and evidence intelligence.",
  },
  {
    title: "Police & Investigation Agencies",
    href: "/solutions/investigation-agencies",
    description: "Recognition, comparison, intelligence gathering, and report preparation for investigation teams.",
  },
  {
    title: "Digital Forensics",
    href: "/solutions/digital-forensics",
    description: "Support media authenticity, OSINT, device-related evidence, and structured case intelligence.",
  },
  {
    title: "Fraud & Identity Teams",
    href: "/solutions/fraud-identity",
    description: "Face matching, liveness checks, manipulated media detection, and suspicious identity review.",
  },
  {
    title: "Research & Training Institutes",
    href: "/solutions/research-training",
    description: "Use forensic AI tools for education, simulation, demonstrations, and research workflows.",
  },
  {
    title: "Enterprise Security",
    href: "/solutions/enterprise-security",
    description: "Secure biometric and intelligence workflows for enterprise risk and verification teams.",
  },
];

export const resourceLinks = [
  {
    title: "Documentation",
    href: "/resources/docs",
    description: "Guides for using NexGen Forensics products, APIs, and workflow modules.",
  },
  {
    title: "API Reference",
    href: "/resources/api-reference",
    description: "Technical API endpoints, request examples, authentication, and integration notes.",
  },
  {
    title: "Case Studies",
    href: "/resources/case-studies",
    description: "Example workflows showing forensic AI use in realistic investigation scenarios.",
  },
  {
    title: "Research Notes",
    href: "/resources/research",
    description: "Technical notes about biometrics, media authenticity, fingerprint AI, OSINT, and forensic intelligence.",
  },
  {
    title: "Validation & Accuracy",
    href: "/resources/validation",
    description: "Information about testing, model evaluation, performance boundaries, and responsible use.",
  },
  {
    title: "Blog / Insights",
    href: "/resources/insights",
    description: "Articles about forensic AI, digital investigation, biometric intelligence, and product updates.",
  },
];

export const demoLinks = [
  { title: "iMatch Face Recognition Demo", href: "/demo/imatch" },
  { title: "Fingerprint AI Demo", href: "/demo/fingerprint-ai" },
  { title: "Deepfake Detection Demo", href: "/demo/deepfake-detection" },
  { title: "OSINT Intelligence Demo", href: "/demo/osint" },
  { title: "Evidence Graph Demo", href: "/demo/evidence-graph" },
];

export const navGroups = {
  products: {
    label: "Products",
    title: "Products",
    subtitle:
      "AI-powered forensic intelligence products for investigation, recognition, analysis, and reporting workflows.",
    href: "/products",
    items: productLinks,
    featured: {
      title: "Product Suite",
      text: "Connect face recognition, fingerprint AI, deepfake detection, OSINT, and evidence intelligence in one forensic-grade ecosystem.",
      cta: "Explore All Products",
      href: "/products",
    },
  },
  solutions: {
    label: "Solutions",
    title: "Solutions",
    subtitle: "Built for forensic teams, investigation units, institutions, and enterprise security workflows.",
    href: "/solutions",
    items: solutionLinks,
    featured: {
      title: "Find the Right NexGen Workflow",
      text: "Choose a solution based on your team type, evidence workflow, and forensic AI requirements.",
      cta: "View Solutions",
      href: "/solutions",
    },
  },
  resources: {
    label: "Resources",
    title: "Resources",
    subtitle: "Learn, validate, and explore forensic AI workflows.",
    href: "/resources",
    items: resourceLinks,
    featured: {
      title: "Responsible Forensic AI",
      text: "NexGen Forensics is designed to assist expert review, not replace professional judgment.",
      cta: "Read Validation Notes",
      href: "/resources/validation",
    },
  },
};
