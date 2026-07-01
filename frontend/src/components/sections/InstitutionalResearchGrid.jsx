import { motion } from "framer-motion";
import { research } from "../../constants/data";
import "./InstitutionalResearchGrid.css";

/**
 * Chapter Six: Research that reads like an institution.
 * Academic grid displaying different operational/forensic intelligence research domains.
 */
export function InstitutionalResearchGrid() {
  return (
    <section className="nx-research">
      <div className="nx-section-heading">
        <p className="nx-kicker">Chapter Six</p>
        <h2>Research that reads like an institution, not content marketing.</h2>
      </div>
      <div className="nx-research-grid">
        {research.map((item, index) => (
          <motion.article
            key={item.title}
            className="nx-research-card"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.04, duration: 0.5 }}
          >
            <span>{String(index + 1).padStart(2, "0")}</span>
            <h3>{item.title}</h3>
            <p>{item.desc}</p>
          </motion.article>
        ))}
      </div>
    </section>
  );
}

export default InstitutionalResearchGrid;
