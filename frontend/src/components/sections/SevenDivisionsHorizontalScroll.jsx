import { useRef, useState } from "react";
import {
  motion,
  useScroll,
  useTransform,
  useSpring,
  useMotionTemplate,
  useMotionValueEvent,
} from "framer-motion";
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

  // Scroll position picks a slide; a spring carries the track there. Snapping
  // to whole slides means the deck never rests half-way with an image cut off
  // at the viewport edge, and every transition plays as one fluid glide
  // instead of tracking the scrollbar 1:1.
  const targetX = useTransform(scrollYProgress, (progress) => {
    const index = Math.min(
      divisions.length - 1,
      Math.max(0, Math.round(progress * (divisions.length - 1)))
    );
    return -index * 100;
  });
  const springX = useSpring(targetX, { stiffness: 115, damping: 26, mass: 0.75 });
  const x = useMotionTemplate`${springX}%`;

  const [active, setActive] = useState(0);

  // Sync current slide index with active dots indicator
  useMotionValueEvent(targetX, "change", (latest) => {
    setActive(Math.round(-latest / 100));
  });


  return (
    <section className="nx-platform-scroll" ref={containerRef}>
      <div className="nx-platform-sticky">
        <div className="nx-platform-header">
          <h2>Explore our product</h2>
        </div>
        <div className="nx-platform-footer">
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
          {divisions.map((division, index) => (
            <article className="nx-division-panel" key={division.id}>
              <a
                className="nx-image-container"
                href={division.href}
                aria-label={`Open the ${division.title} product page`}
              >
                <img
                  src={divisionImages[division.id]}
                  alt={division.title}
                  className="nx-division-image"
                  loading={index === 0 ? "eager" : "lazy"}
                  decoding="async"
                />
              </a>
            </article>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

export default SevenDivisionsHorizontalScroll;
