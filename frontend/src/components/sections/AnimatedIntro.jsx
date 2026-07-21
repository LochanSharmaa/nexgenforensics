import { motion } from "framer-motion";
import introPoster from "../../assets/nexgen-hero.jfif";
import introVideo from "../../assets/nexgen-hero.mp4";
import "./AnimatedIntro.css";

export function AnimatedIntro({ onExplore }) {
  return (
    <motion.section
      className="nx-intro"
      aria-label="NexGen Forensics introduction"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.015 }}
      transition={{ duration: 0.6 }}
    >
      <img className="nx-intro-blur" src={introPoster} alt="" aria-hidden="true" />
      <video
        className="nx-intro-video"
        src={introVideo}
        poster={introPoster}
        autoPlay
        loop
        muted
        playsInline
        preload="auto"
        aria-label="Abstract rose and pearl forensic technology animation"
      />

      <div className="nx-intro-shade" aria-hidden="true" />

      <div className="nx-intro-content">
        <p className="nx-intro-kicker">Intelligence beyond evidence</p>

        <div className="nx-intro-main">
          <h1>
            NexGen
            <span>Forensics</span>
          </h1>
          <p>Advanced forensic technology designed to reveal what others leave unseen.</p>
        </div>

        <div className="nx-intro-action">
          <button type="button" onClick={onExplore}>
            <span>Explore Our Products</span>
            <span className="nx-intro-arrow" aria-hidden="true">→</span>
          </button>
          <small>Enter the platform</small>
        </div>
      </div>
    </motion.section>
  );
}

export default AnimatedIntro;
