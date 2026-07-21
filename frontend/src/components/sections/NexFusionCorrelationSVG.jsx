import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { correlationSources } from "../../constants/data";
import "./NexFusionCorrelationSVG.css";

/**
 * Chapter Two: Intelligence is not collected. It is connected.
 * Interactive SVG rendering connector paths and source cards that converge into NexFusion model block.
 */
export function NexFusionCorrelationSVG() {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  });

  const pathLength = useTransform(scrollYProgress, [0.25, 0.72], [0, 1]);
  const modelOpacity = useTransform(scrollYProgress, [0.55, 0.75], [0, 1]);
  const modelScale = useTransform(scrollYProgress, [0.55, 0.75], [0.92, 1]);
  const copyY = useTransform(scrollYProgress, [0.15, 0.72], [80, -40]);

  return (
    <section className="nx-correlation-scene" ref={containerRef}>
      <div className="nx-correlation-stage">
        <motion.div className="nx-correlation-copy" style={{ y: copyY }}>
          <p className="nx-kicker">Chapter Two</p>
          <h2>Intelligence is not collected. It is connected.</h2>
          <p>
            NexFusion turns isolated findings into a single operational model built for
            investigative decisions.
          </p>
        </motion.div>

        <svg className="nx-correlation-svg" viewBox="0 0 1400 720" aria-hidden="true">
          <defs>
            <filter id="paperShadow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="12" stdDeviation="18" floodOpacity="0.08" />
            </filter>
          </defs>
          {[60, 210, 360, 510].map((y, index) => (
            <motion.path
              key={y}
              d={`M 400 ${y + 40} C 560 ${y - 60} 750 ${360 + index * 18} 920 360`}
              className="nx-link-path"
              style={{ pathLength }}
            />
          ))}
          {correlationSources.map((source, index) => (
            <motion.g
              key={source.label}
              initial={{ opacity: 0, x: -18 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-15%" }}
              transition={{ delay: index * 0.08, duration: 0.55 }}
              filter="url(#paperShadow)"
            >
              <rect x="32" y={60 + index * 150} width="370" height="88" rx="18" className="nx-svg-card" />
              <text x="72" y={103 + index * 150} className="nx-svg-label">
                {source.label}
              </text>
              <text x="72" y={127 + index * 150} className="nx-svg-meta">
                {source.meta}
              </text>
            </motion.g>
          ))}
          <motion.g style={{ opacity: modelOpacity, scale: modelScale }} filter="url(#paperShadow)">
            <rect x="892" y="232" width="400" height="256" rx="28" className="nx-svg-core" />
            <text x="934" y="306" className="nx-svg-core-label">
              NexFusion
            </text>
            <text x="934" y="348" className="nx-svg-core-title">
              Unified Case Model
            </text>
            <text x="934" y="392" className="nx-svg-meta nx-svg-meta-light">
              Entity resolution
            </text>
            <text x="934" y="423" className="nx-svg-meta nx-svg-meta-light">
              Relationship confidence
            </text>
          </motion.g>
        </svg>
      </div>
    </section>
  );
}

export default NexFusionCorrelationSVG;
