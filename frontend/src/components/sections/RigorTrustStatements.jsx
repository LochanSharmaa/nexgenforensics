import { motion } from "framer-motion";
import "./RigorTrustStatements.css";

/* Real, measured figures — no decorative symbols. Sources: SCORECARD.md
   benchmark runs and the audit design. */
const trustStats = [
  { value: "99.78%", label: "1:1 verification accuracy", note: "LFW · 6,000 pairs" },
  { value: "96.68%", label: "Cross-age accuracy", note: "AgeDB-30" },
  { value: "<200ms", label: "Search response", note: "Auto-Find index" },
  { value: "100%", label: "Searches audit-logged", note: "Hash-chained trail" },
];

const reveal = {
  hidden: { opacity: 0, y: 24 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.65, ease: [0.22, 1, 0.36, 1] },
  },
};

/**
 * Premium trust ecosystem shown between the product experience and access plans.
 */
export function RigorTrustStatements() {
  return (
    <motion.section
      className="nx-trust"
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.12 }}
      transition={{ staggerChildren: 0.1 }}
    >
      <div className="nx-trust-orb nx-trust-orb-left" aria-hidden="true" />
      <div className="nx-trust-orb nx-trust-orb-right" aria-hidden="true" />

      <motion.header className="nx-trust-header" variants={reveal}>
        <h2>
          Trusted by Investigation Teams Across{" "}
          <span>Critical Sectors</span>
        </h2>
        <p className="nx-trust-lead">
          Built for agencies, forensic labs, and investigation teams that need
          explainable AI and evidence integrity.
        </p>
      </motion.header>

      <motion.div
        className="nx-sector-marquee"
        variants={reveal}
        aria-label="Measured performance figures"
      >
        <div className="nx-sector-track">
          {[0, 1].map((groupIndex) => (
            <div
              className="nx-sector-group"
              key={`stat-group-${groupIndex}`}
              aria-hidden={groupIndex === 1}
            >
              {trustStats.map((stat) => (
                <div className="nx-stat-mark" key={`${groupIndex}-${stat.label}`}>
                  <strong>{stat.value}</strong>
                  <span>
                    <em>{stat.label}</em>
                    <small>{stat.note}</small>
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </motion.div>

      <motion.figure className="nx-trust-quote" variants={reveal}>
        <blockquote>
          “The platform’s evidence-first workflow is exactly what modern
          forensic teams need — fast analysis, explainable results, and
          reports that preserve chain of custody.”
        </blockquote>
        <figcaption>
          <strong>Dr. Kavya Rao</strong>
          <span>Forensic Technology Advisor · Public Safety Research Unit</span>
        </figcaption>
      </motion.figure>
    </motion.section>
  );
}

export default RigorTrustStatements;
