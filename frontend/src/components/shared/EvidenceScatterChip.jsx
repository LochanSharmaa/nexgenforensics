import { motion, useTransform } from "framer-motion";
import "./EvidenceScatterChip.css";

/**
 * An individual evidence chip that animates its position (x, y) and opacity
 * based on the parent section's scroll progress.
 * 
 * @param {Object} props
 * @param {string} props.item - The evidence name/text label.
 * @param {number} props.index - Index of the chip in the array.
 * @param {number} props.total - Total number of chips.
 * @param {import("framer-motion").MotionValue<number>} props.scrollYProgress - Progress value of scroll.
 */
export function EvidenceScatterChip({ item, index, total, scrollYProgress }) {
  const angle = (index / total) * Math.PI * 2 - Math.PI / 2;

  const farRadiusX = 42 + (index % 4) * 4;
  const farRadiusY = 34 + (index % 4) * 3;

  const orbitRadiusX = 31 + (index % 2) * 7;
  const orbitRadiusY = 24 + (index % 2) * 5;

  const finalRadiusX = 25 + (index % 2) * 8;
  const finalRadiusY = 19 + (index % 2) * 5;

  const farX = Math.cos(angle) * farRadiusX;
  const farY = Math.sin(angle) * farRadiusY;

  const orbitX = Math.cos(angle) * orbitRadiusX;
  const orbitY = Math.sin(angle) * orbitRadiusY;

  const finalX = Math.cos(angle) * finalRadiusX;
  const finalY = Math.sin(angle) * finalRadiusY;

  const appearStart = 0.03 + index * 0.01;

  const itemOpacity = useTransform(
    scrollYProgress,
    [appearStart, appearStart + 0.05, 0.85],
    [0, 1, 1]
  );

  const itemX = useTransform(
    scrollYProgress,
    [0.06, 0.32, 0.58],
    [`${farX}vw`, `${orbitX}vw`, `${finalX}vw`]
  );

  const itemY = useTransform(
    scrollYProgress,
    [0.06, 0.32, 0.58],
    [`${farY}vh`, `${orbitY}vh`, `${finalY}vh`]
  );

  const itemScale = useTransform(
    scrollYProgress,
    [0.03, 0.28, 0.58],
    [0.88, 1, 0.96]
  );

  return (
    <motion.div
      className="nx-evidence-chip"
      style={{ opacity: itemOpacity, x: itemX, y: itemY, scale: itemScale }}
    >
      <span>{String(index + 1).padStart(2, "0")}</span>
      {item}
    </motion.div>
  );
}

export default EvidenceScatterChip;
