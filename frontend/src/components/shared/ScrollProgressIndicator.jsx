import { motion, useScroll, useSpring } from "framer-motion";
import "./ScrollProgressIndicator.css";

/**
 * A thin fixed-position progress bar at the top of the viewport
 * that fills as the page is scrolled down.
 */
export function ScrollProgressIndicator() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 28, mass: 0.2 });

  return <motion.div className="nx-scroll-progress" style={{ scaleX }} />;
}

export default ScrollProgressIndicator;
