import { useRef, useState } from "react";
import { motion, useScroll, useTransform, useMotionValueEvent } from "framer-motion";
import { divisions } from "../../constants/data";
import "./SevenDivisionsHorizontalScroll.css";

// Import product division showcase images
import faceSearchImg from "../../assets/Face_Search.jpeg";
import videoAnalysisImg from "../../assets/Video_Analysis.jpeg";
import osintImg from "../../assets/OSINT.jpeg";
import deepfakeDetectionImg from "../../assets/Deepfake_detection.jpeg";
import crime3DImg from "../../assets/3D_crime.jpeg";
import evidenceGraphImg from "../../assets/Evidence_Graph.jpeg";
import caseIntelligenceImg from "../../assets/Case_Intelligence.jpeg";

const divisionImages = {
  identity: faceSearchImg,
  digital: videoAnalysisImg,
  osint: osintImg,
  audio: deepfakeDetectionImg,
  document: crime3DImg,
  fusion: evidenceGraphImg,
  command: caseIntelligenceImg,
};

/**
 * Chapter Three: Explore our product
 * A horizontal scroll-triggered slide deck revealing different intelligence division product interfaces.
 */
export function SevenDivisionsHorizontalScroll() {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  // Slide track translation based on page scrolling progress
  const x = useTransform(scrollYProgress, [0, 1], ["0%", `-${(divisions.length - 1) * 100}%`]);
  const [active, setActive] = useState(0);

  // Sync current slide index with active dots indicator
  useMotionValueEvent(scrollYProgress, "change", (latest) => {
    const next = Math.min(
      divisions.length - 1,
      Math.max(0, Math.round(latest * (divisions.length - 1)))
    );
    setActive(next);
  });


  return (
    <section className="nx-platform-scroll" ref={containerRef}>
      <div className="nx-platform-sticky">
        <div className="nx-platform-header">
          <h2>Explore our product</h2>
        </div>
        <div className="nx-platform-footer">
          <p className="nx-kicker">Chapter Three</p>
          <div className="nx-platform-index">
            {divisions.map((division, index) => (
              <span
                key={division.id}
                className={index === active ? "active" : ""}
                title={`Go to Division ${index + 1}`}
              >
                {String(index + 1).padStart(2, "0")}
              </span>
            ))}
          </div>
        </div>
        <motion.div className="nx-platform-track" style={{ x }}>
          {divisions.map((division) => (
            <article className="nx-division-panel" key={division.id}>
              <div className="nx-image-container">
                <img
                  src={divisionImages[division.id]}
                  alt={division.title}
                  className="nx-division-image"
                />
              </div>
            </article>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

export default SevenDivisionsHorizontalScroll;
