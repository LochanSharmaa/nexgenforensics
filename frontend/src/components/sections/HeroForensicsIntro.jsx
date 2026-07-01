import introPoster from "../../assets/nexgen-hero.jfif";
import introVideo from "../../assets/nexgen-hero.mp4";
import "./HeroForensicsIntro.css";

/**
 * Main landing page hero using the NexGen animated identity.
 */
export function HeroForensicsIntro() {
  return (
    <section className="nx-hero" id="top">
      <img className="nx-hero-blur" src={introPoster} alt="" aria-hidden="true" />
      <video
        className="nx-hero-video"
        src={introVideo}
        poster={introPoster}
        autoPlay
        loop
        muted
        playsInline
        preload="auto"
        aria-label="Abstract rose and pearl forensic technology animation"
      />
      <div className="nx-hero-shade" aria-hidden="true" />

      <div className="nx-hero-content">
        <p className="nx-hero-kicker">Intelligence beyond evidence</p>

        <div className="nx-hero-main">
          <h1>
            NexGen
            <span>Forensics</span>
          </h1>
          <p>Advanced forensic technology designed to reveal what others leave unseen.</p>
        </div>

        <div className="nx-hero-action">
          <a href="#platform">
            <span>Explore Our Products</span>
            <span className="nx-hero-arrow" aria-hidden="true">→</span>
          </a>
          <small>Enter the platform</small>
        </div>
      </div>
    </section>
  );
}

export default HeroForensicsIntro;
