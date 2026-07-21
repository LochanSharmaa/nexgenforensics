import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { evidenceItems } from "../../constants/data";
import { EvidenceScatterChip } from "../shared/EvidenceScatterChip";
import "./EvidenceScatteringScroll.css";

/**
 * Chapter One: The Fragment.
 * An interactive scroll section showing floating evidence chips centering into a resolved case core.
 */
export function EvidenceScatteringScroll() {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  });

  const density = useTransform(scrollYProgress, [0.04, 0.34], [0, 1]);
  const fieldOpacity = useTransform(
    density,
    [0, 1],
    [0.45, 1]
  );
  
  const centerScale = useTransform(scrollYProgress, [0.38, 0.56, 0.78], [0.86, 1.05, 1]);
  const centerOpacity = useTransform(scrollYProgress, [0.34, 0.46], [0, 1]);

  return (
    <section className="nx-evidence-scene" ref={containerRef}>
      <div className="nx-sticky">
        <motion.div className="nx-evidence-field" style={{ opacity: fieldOpacity }}>
          {evidenceItems.map((item, index) => (
            <EvidenceScatterChip
              key={item}
              item={item}
              index={index}
              total={evidenceItems.length}
              scrollYProgress={scrollYProgress}
            />
          ))}

          <motion.div className="nx-case-core" style={{ opacity: centerOpacity, scale: centerScale }}>
            <span>EVIDENCE CONNECTED</span>
            <strong>CASE FILE READY</strong>
          </motion.div>
        </motion.div>

        <div className="nx-scene-copy nx-scene-copy-left">
          <p className="nx-kicker">Chapter One</p>
          <h2>Evidence arrives fragmented.</h2>
          <p>
            NexGen transforms scattered forensic signals — faces, fingerprints, CCTV, vehicles,
            phones, locations, OSINT traces, documents, and media — into one structured
            intelligence picture investigators can trust.
          </p>
        </div>
      </div>
    </section>
  );
}

export default EvidenceScatteringScroll;
