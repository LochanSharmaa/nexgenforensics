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
        aria-label="Abstract biometric enterprise technology animation"
      />
      <div className="nx-hero-shade" aria-hidden="true" />

      <div className="nx-hero-content">
        <p className="nx-hero-kicker">Commercial biometric AI platform</p>

        <div className="nx-hero-main">
          <h1>NexGen <span>Identity</span></h1>
          <p>
            Enterprise facial recognition for identity verification, secure face search,
            fraud prevention, and privacy-aware biometric deployment.
          </p>
        </div>

        <div className="nx-hero-action">
          <a href="#platform">
            <span>Explore Platform</span>
            <span className="nx-hero-arrow" aria-hidden="true">→</span>
          </a>
          <small>Target accuracy is benchmarked before commercial claims</small>
        </div>
      </div>
    </section>
  );
}

export default HeroForensicsIntro;
