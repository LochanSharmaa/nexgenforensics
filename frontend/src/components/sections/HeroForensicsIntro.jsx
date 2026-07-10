import { useLayoutEffect, useRef } from "react";
import gsap from "gsap";
import forensicHand from "../../assets/forensic-hand-cutout.png";
import introPoster from "../../assets/nexgen-hero.jfif";
import introVideo from "../../assets/nexgen-hero.mp4";
import "./HeroForensicsIntro.css";

/**
 * Main landing page hero using the NexGen animated identity.
 */
export function HeroForensicsIntro() {
  const heroRef = useRef(null);
  const handRef = useRef(null);
  const wordRef = useRef(null);
  const frameRef = useRef(null);
  const scanRef = useRef(null);

  useLayoutEffect(() => {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reduceMotion) {
      gsap.set([handRef.current, frameRef.current], { autoAlpha: 1 });
      gsap.set(scanRef.current, { autoAlpha: 0 });
      return undefined;
    }

    const ctx = gsap.context(() => {
      const hand = handRef.current;
      const word = wordRef.current;
      const frame = frameRef.current;
      const scan = scanRef.current;

      gsap.set(hand, {
        autoAlpha: 0.95,
        xPercent: -50,
        yPercent: -50,
        x: 64,
        y: 54,
        rotate: 4,
        scale: 0.98,
        transformOrigin: "54% 60%",
      });
      gsap.set(frame, { autoAlpha: 0 });
      gsap.set(scan, { autoAlpha: 0, xPercent: -120, scaleX: 0.72, transformOrigin: "left center" });

      const tl = gsap.timeline({
        defaults: { ease: "power3.out" },
        delay: 0.55,
      });

      // Animation speed: change these durations for a slower or tighter capture.
      tl.to(hand, {
        x: -8,
        y: -8,
        rotate: -1.4,
        scale: 1.015,
        duration: 2.35,
        ease: "expo.out",
      })
        .to(frame, { autoAlpha: 1, duration: 0.45 }, "-=1.12")
        .to(scan, { autoAlpha: 1, duration: 0.16 }, "-=0.74")
        .to(scan, { xPercent: 120, scaleX: 1, duration: 0.92, ease: "power3.out" }, "<")
        .to(scan, { autoAlpha: 0, duration: 0.22 }, "-=0.12")
        .to(
          word,
          {
            y: -12,
            scale: 1.025,
            color: "#211719",
            textShadow: "0 14px 34px rgba(33, 23, 25, 0.16)",
            duration: 0.72,
            ease: "power3.out",
          },
          "-=0.58",
        )
        .to(
          hand,
          {
            x: -17,
            y: -14,
            rotate: -3.2,
            scale: 0.992,
            duration: 0.32,
            ease: "power3.out",
          },
          "-=0.56",
        )
        .to(hand, { x: -12, y: -10, rotate: -1.7, scale: 1, duration: 0.84, ease: "back.out(1.12)" })
        .to(
          word,
          {
            y: 0,
            scale: 1,
            color: "#211719",
            textShadow: "0 8px 26px rgba(255, 255, 255, 0.62)",
            duration: 0.95,
            ease: "expo.out",
          },
          "-=0.68",
        )
        .to(frame, { autoAlpha: 0.78, duration: 0.8, ease: "power3.out" }, "-=0.5");
    }, heroRef);

    return () => ctx.revert();
  }, []);

  return (
    <section className="nx-hero" id="top" ref={heroRef}>
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
        <p className="nx-hero-kicker">AI forensic examination platform</p>

        <div className="nx-hero-main">
          {/* Target word: change this text and the matching span to capture a different word. */}
          <h1 className="nx-hero-title">
            <span>Automated</span>{" "}
            <span className="nx-capture-wrap">
              <img className="nx-capture-hand" ref={handRef} src={forensicHand} alt="" aria-hidden="true" />
              <span className="nx-capture-target" ref={wordRef}>
                Forensic
              </span>
              <span className="nx-capture-frame" ref={frameRef} aria-hidden="true">
                <span className="nx-corner nx-corner-tl" />
                <span className="nx-corner nx-corner-tr" />
                <span className="nx-corner nx-corner-bl" />
                <span className="nx-corner nx-corner-br" />
              </span>
              <span className="nx-capture-scan" ref={scanRef} aria-hidden="true" />
            </span>{" "}
            <span>Examination</span>
          </h1>
          <p>
            Automated evidence inspection, biometric matching, and case intelligence
            built for defensible forensic workflows.
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
