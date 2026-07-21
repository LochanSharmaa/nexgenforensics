import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import "./FutureOfForensicsQuote.css";

/**
 * Centered futuristic statement about the future of forensic science
 * that scales slightly as it is scrolled into view.
 */
export function FutureOfForensicsQuote() {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  });

  const scale = useTransform(scrollYProgress, [0.1, 0.7], [0.88, 1.0]);

  return (
    <section className="nx-future" ref={containerRef}>
      <motion.h2 style={{ scale }}>
        The future of forensic intelligence will not be defined by more information.
        <br />
        <span>It will be defined by better understanding.</span>
      </motion.h2>
    </section>
  );
}

export default FutureOfForensicsQuote;
