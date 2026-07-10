export const productLinks = [
  {
    title: "NexGen iMatch",
    category: "Enterprise Facial Recognition",
    href: "/products/imatch",
    description:
      "Face search, face matching, liveness checks, deepfake detection, and secure tenant-isolated biometric workflows.",
  },
  {
    title: "NexGen Fingerprint AI",
    category: "AI Fingerprint Examination Platform",
    href: "/products/fingerprint-ai",
    description:
      "Biometric quality tooling for controlled enrollment, identity review, and secure customer verification workflows.",
  },
  {
    title: "Deepfake Detection",
    category: "Synthetic Media Analysis",
    href: "/products/deepfake-detection",
    description:
      "Detect manipulated facial media, synthetic images, AI-generated videos, and authenticity risk signals.",
  },
  {
    title: "Identity Intelligence",
    category: "Enterprise Identity Platform",
    href: "/products/osint",
    description:
      "Organize identity signals, workspace records, and permitted metadata for verification workflows.",
  },
  {
    title: "Tenant Graph",
    category: "Workspace Relationship Analysis",
    href: "/products/evidence-graph",
    description:
      "Visualize relationships between customers, accounts, devices, locations, and permissioned identity signals.",
  },
  {
    title: "Review Intelligence",
    category: "AI Review Management",
    href: "/products/case-intelligence",
    description:
      "Summarize review materials, structure analyst notes, manage decision context, and support human approval workflows.",
  },
  {
    title: "API Platform",
    category: "Biometric AI API",
    href: "/products/api",
    description:
      "Integrate NexGen recognition, liveness, search, and audit APIs into secure enterprise systems.",
  },
];

export const solutionLinks = [
  {
    title: "Fintech KYC",
    href: "/solutions/fintech-kyc",
    description: "Face verification, liveness checks, and fraud-review workflows for regulated onboarding teams.",
  },
  {
    title: "Corporate Access",
    href: "/solutions/corporate-access",
    description: "Secure facility authentication, workforce verification, and tenant-controlled access policies.",
  },
  {
    title: "Fraud Operations",
    href: "/solutions/fraud-operations",
    description: "Support suspicious identity review, manipulated media checks, and structured decision queues.",
  },
  {
    title: "Fraud & Identity Teams",
    href: "/solutions/fraud-identity",
    description: "Face matching, liveness checks, manipulated media detection, and suspicious identity review.",
  },
  {
    title: "Research & Validation Teams",
    href: "/solutions/research-validation",
    description: "Use biometric AI tools for benchmark testing, model evaluation, and readiness review.",
  },
  {
    title: "Enterprise Security",
    href: "/solutions/enterprise-security",
    description: "Secure biometric workflows for enterprise risk, workforce identity, and customer verification teams.",
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
    title: "Customer Scenarios",
    href: "/resources/case-studies",
    description: "Example workflows showing biometric AI use in realistic commercial deployment scenarios.",
  },
  {
    title: "Research Notes",
    href: "/resources/research",
    description: "Technical notes about biometrics, media authenticity, tenant isolation, and identity intelligence.",
  },
  {
    title: "Validation & Accuracy",
    href: "/resources/validation",
    description: "Information about testing, model evaluation, performance boundaries, and responsible use.",
  },
  {
    title: "Blog / Insights",
    href: "/resources/insights",
    description: "Articles about biometric AI, identity verification, model governance, and product updates.",
  },
];

export const demoLinks = [
  { title: "iMatch Face Recognition Demo", href: "/demo/imatch" },
  { title: "Biometric Quality Demo", href: "/demo/fingerprint-ai" },
  { title: "Deepfake Detection Demo", href: "/demo/deepfake-detection" },
  { title: "Identity Intelligence Demo", href: "/demo/osint" },
  { title: "Tenant Graph Demo", href: "/demo/evidence-graph" },
];

export const navGroups = {
  products: {
    label: "Products",
    title: "Products",
    subtitle:
      "AI-powered biometric products for enterprise recognition, verification, search, and reporting workflows.",
    href: "/products",
    items: productLinks,
    featured: {
      title: "Product Suite",
      text: "Connect face recognition, liveness, deepfake detection, tenant search, and audit intelligence in one enterprise-grade ecosystem.",
      cta: "Explore All Products",
      href: "/products",
    },
  },
  solutions: {
    label: "Solutions",
    title: "Solutions",
    subtitle: "Built for commercial identity teams, fraud operations, access control, and enterprise security workflows.",
    href: "/solutions",
    items: solutionLinks,
    featured: {
      title: "Find the Right NexGen Workflow",
      text: "Choose a solution based on your team type, identity workflow, risk model, and deployment requirements.",
      cta: "View Solutions",
      href: "/solutions",
    },
  },
  resources: {
    label: "Resources",
    title: "Resources",
    subtitle: "Learn, validate, and explore commercial biometric AI workflows.",
    href: "/resources",
    items: resourceLinks,
    featured: {
      title: "Responsible Biometric AI",
      text: "NexGen Forensics is designed to assist expert review, not replace professional judgment.",
      cta: "Read Validation Notes",
      href: "/resources/validation",
    },
  },
};
