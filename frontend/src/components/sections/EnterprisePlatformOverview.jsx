import { motion } from "framer-motion";
import "./EnterprisePlatformOverview.css";

const platformPillars = [
  {
    title: "Consent-Aware Intake",
    metric: "01",
    copy: "Client uploads are checked for purpose, permission, image quality, retention policy, and workspace ownership before biometric processing begins.",
    items: ["Permission validation", "Quality rejection reasons", "Retention controls"],
  },
  {
    title: "Auto-Find Search",
    metric: "<200ms",
    copy: "A single authorized face image can search the permitted enterprise index and return ranked matches with confidence, metadata, and operator context.",
    items: ["Workspace index", "Ranked candidates", "Decision support notes"],
  },
  {
    title: "Controlled Improvement",
    metric: "A/B",
    copy: "Fine-tuning jobs are queued only after validation and deploy only when documented benchmark gates improve with rollback available.",
    items: ["Benchmark target", "Versioned deployment", "Rollback ready"],
  },
  {
    title: "Compliance-Ready Controls",
    metric: "24/7",
    copy: "Every search, enrollment, deletion, model update, and permission change is written to an audit trail designed for enterprise review.",
    items: ["RBAC", "Template encryption", "Audit exports"],
  },
];

const tenantRows = [
  ["Fintech KYC", "Face verification", "Fraud review", "Enabled"],
  ["Corporate Access", "Entry authentication", "Liveness check", "Workspace only"],
  ["Workforce Identity", "Employee verification", "Retention policy", "Disabled"],
  ["Retail Risk", "Loss-prevention review", "Human approval", "Legal agreement required"],
];

export function EnterprisePlatformOverview() {
  return (
    <section className="nx-enterprise" id="enterprise-platform" aria-labelledby="enterprise-title">
      <div className="nx-enterprise-header">
        <p className="nx-kicker">Enterprise Platform</p>
        <h2 id="enterprise-title">Built for sellable commercial biometric deployment.</h2>
        <p>
          NexGen Forensics is positioned for private-sector identity verification,
          fraud prevention, access control, and secure face search. The 99.99%
          accuracy figure is treated as a benchmark goal until independently validated.
        </p>
      </div>

      <div className="nx-enterprise-grid">
        {platformPillars.map((pillar, index) => (
          <motion.article
            className="nx-enterprise-card"
            key={pillar.title}
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.25 }}
            transition={{ delay: index * 0.08, duration: 0.5 }}
          >
            <span>{pillar.metric}</span>
            <h3>{pillar.title}</h3>
            <p>{pillar.copy}</p>
            <ul>
              {pillar.items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </motion.article>
        ))}
      </div>

      <div className="nx-tenant-console" aria-label="Commercial tenant configuration preview">
        <div className="nx-tenant-topbar">
          <span>Tenant Controls</span>
          <strong>Cross-client search disabled by default</strong>
        </div>
        <div className="nx-tenant-table">
          <div className="nx-tenant-head">
            <span>Client Type</span>
            <span>Workflow</span>
            <span>Control</span>
            <span>Cross-Client</span>
          </div>
          {tenantRows.map((row) => (
            <div className="nx-tenant-row" key={row.join("-")}>
              {row.map((cell) => (
                <span key={cell}>{cell}</span>
              ))}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default EnterprisePlatformOverview;
