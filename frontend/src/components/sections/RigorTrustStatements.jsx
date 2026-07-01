import { motion } from "framer-motion";
import kavyaAvatar from "../../assets/kavya.jpg";
import manAvatar from "../../assets/man.jpg";
import sectorSprite from "../../assets/sector-sprite-ai.png";
import "./RigorTrustStatements.css";

const trustedSectors = [
  { title: "Law Enforcement", col: 0, row: 0 },
  { title: "Digital Forensics Labs", col: 1, row: 0 },
  { title: "National Security", col: 2, row: 0 },
  { title: "Cyber Crime Units", col: 3, row: 0 },
  { title: "Enterprise Risk", col: 4, row: 0 },
  { title: "Border Security", col: 0, row: 1 },
  { title: "Fraud Investigation", col: 1, row: 1 },
  { title: "Intelligence Teams", col: 2, row: 1 },
  { title: "Research Partners", col: 3, row: 1 },
  { title: "Public Safety", col: 4, row: 1 },
];

const testimonials = [
  {
    quote:
      "NexGen gives investigators a single operational view across faces, fingerprints, video, OSINT, documents, and devices. It turns fragmented evidence into a structured case picture.",
    name: "Aarav Mehta",
    role: "Digital Forensics Lead",
    organization: "Regional Investigation Lab",
    initials: "AM",
    avatar: manAvatar,
    label: "Verified Workflow",
  },
  {
    quote:
      "The platform’s evidence-first workflow is exactly what modern forensic teams need — fast analysis, explainable results, and reports that preserve chain of custody.",
    name: "Dr. Kavya Rao",
    role: "Forensic Technology Advisor",
    organization: "Public Safety Research Unit",
    initials: "KR",
    avatar: kavyaAvatar,
    label: "Evidence-Grade",
  },
];

const trustStats = [
  { value: "8", label: "AI forensic products" },
  { value: "✓", label: "Evidence-grade workflows" },
  { value: "∞", label: "Chain-of-custody logging" },
  { value: "01", label: "Report-ready outputs" },
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
        <p className="nx-trust-pill">Trusted Ecosystem</p>
        <h2>
          Trusted by Investigation Teams Across{" "}
          <span>Critical Sectors</span>
        </h2>
        <p className="nx-trust-lead">
          NexGen Forensics is designed for agencies, forensic labs, enterprises,
          and investigation teams that need explainable AI, evidence integrity,
          and operational reliability.
        </p>
      </motion.header>

      <motion.div
        className="nx-sector-marquee"
        variants={reveal}
        aria-label="Sectors using NexGen Forensics"
        style={{ "--sector-sprite": `url(${sectorSprite})` }}
      >
        <div className="nx-sector-track">
          {[0, 1].map((groupIndex) => (
            <div
              className="nx-sector-group"
              key={`sector-group-${groupIndex}`}
              aria-hidden={groupIndex === 1}
            >
              {trustedSectors.map((sector) => (
                <div
                  className="nx-sector-mark"
                  key={`${groupIndex}-${sector.title}`}
                >
                  <span
                    className="nx-sector-image"
                    aria-hidden="true"
                    style={{
                      "--sector-x": sector.col,
                      "--sector-y": sector.row,
                    }}
                  />
                  <span className="nx-sector-name">{sector.title}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </motion.div>

      <motion.div
        className="nx-testimonial-grid"
        variants={{
          hidden: {},
          visible: { transition: { staggerChildren: 0.14 } },
        }}
      >
        {testimonials.map((testimonial) => (
          <motion.article
            className="nx-testimonial-card"
            key={testimonial.name}
            variants={reveal}
          >
            <div className="nx-testimonial-topline">
              <span className="nx-quote-mark" aria-hidden="true">
                “
              </span>
              <span className="nx-testimonial-label">
                {testimonial.label}
              </span>
            </div>
            <blockquote>{testimonial.quote}</blockquote>
            <footer className="nx-testimonial-author">
              <img
                className="nx-author-avatar"
                src={testimonial.avatar}
                alt=""
                aria-hidden="true"
              />
              <span>
                <strong>{testimonial.name}</strong>
                <small>{testimonial.role}</small>
                <small>{testimonial.organization}</small>
              </span>
            </footer>
          </motion.article>
        ))}
      </motion.div>

      <motion.div className="nx-trust-stats" variants={reveal}>
        {trustStats.map((stat) => (
          <div className="nx-trust-stat" key={stat.label}>
            <strong>{stat.value}</strong>
            <span>{stat.label}</span>
          </div>
        ))}
      </motion.div>
    </motion.section>
  );
}

export default RigorTrustStatements;
