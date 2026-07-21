import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import "./NexCaseCommandDashboard.css";

/**
 * Chapter Four: Every investigation eventually becomes a decision.
 * Mock UI representational layout of the live investigative dashboard.
 */
export function NexCaseCommandDashboard() {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  });

  const scale = useTransform(scrollYProgress, [0.15, 0.75], [0.92, 1]);
  const y = useTransform(scrollYProgress, [0.1, 0.8], [80, -40]);
  
  const dashboardRows = [
    "Entity timeline",
    "Evidence queue",
    "Persons of interest",
    "Location history",
    "Investigator tasks",
    "Decision record",
  ];

  return (
    <section className="nx-command" ref={containerRef}>
      <div className="nx-command-grid">
        <motion.div className="nx-command-copy" style={{ y }}>
          <p className="nx-kicker">Chapter Four</p>
          <h2>Every investigation eventually becomes a decision.</h2>
          <p>
            NexCase exists for the moment when intelligence must become coordinated operational
            action.
          </p>
        </motion.div>
        
        <motion.div className="nx-command-ui" style={{ scale }}>
          <div className="nx-ui-topbar">
            <span>NEXCASE</span>
            <span>LIVE CASE / NX-0429</span>
          </div>
          <div className="nx-ui-body">
            <div className="nx-ui-side">
              <strong>Case Board</strong>
              {dashboardRows.slice(0, 4).map((row) => (
                <span key={row}>{row}</span>
              ))}
            </div>
            <div className="nx-ui-main">
              <div className="nx-ui-main-header">
                <span>Operational Picture</span>
                <strong>86%</strong>
              </div>
              {dashboardRows.map((row, index) => (
                <motion.div
                  key={row}
                  className="nx-ui-row"
                  initial={{ opacity: 0, y: 18 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-20%" }}
                  transition={{ delay: index * 0.08, duration: 0.45 }}
                >
                  <span>{row}</span>
                  <em className={index % 2 === 0 ? "nx-status-verified" : "nx-status-review"}>
                    {index % 2 === 0 ? "verified" : "review"}
                  </em>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default NexCaseCommandDashboard;
