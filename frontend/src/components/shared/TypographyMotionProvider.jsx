import { useEffect } from "react";
import "./TypographyMotionProvider.css";

// Note: .nx-hero-main h1 is deliberately NOT split — word-splitting flattens
// its inner <span>, losing the black-roman "NexGen" / rose-italic
// "Forensics" pairing. It gets its own CSS entrance instead.
const maskedSelectors = [
  ".nx-scene-copy h2",
  ".nx-correlation-copy h2",
  ".nx-platform-header h2",
  ".nx-command-copy h2",
  ".nx-trust-header h2",
  ".nx-trust-quote blockquote",
  ".nx-section-heading h2",
  ".nx-future h2",
  ".nx-pricing-header h2",
  ".im-hero-copy h1",
  ".im-section-heading h2",
  ".im-final-cta h2",
  ".fp-copy h1",
  ".fp-hand-title",
  ".fp-heading h2",
  ".fp-final h2",
].join(",");

const revealSelectors = [
  ".nx-kicker",
  ".nx-hero-kicker",
  ".nx-hero-main p",
  ".nx-hero-action",
  ".nx-scene-copy p",
  ".nx-correlation-copy p",
  ".nx-command-copy p",
  ".nx-section-heading p",
  ".nx-pricing-lead",
  ".nx-trust-lead",
  ".im-eyebrow",
  ".im-badge",
  ".im-hero-copy p",
  ".im-section-heading p",
  ".im-hero-actions",
  ".im-trust-row",
  ".fp-badge",
  ".fp-kicker",
  ".fp-copy p",
  ".fp-heading p",
  ".fp-actions",
  ".fp-value-row",
].join(",");

const staggerSelectors = [
  ".nx-nav a",
  ".nx-division-panel",
  ".nx-research-card",
  ".nx-pricing-card",
  ".nx-testimonial-card",
  ".nx-trust-stat",
  ".nx-footer-grid a",
  ".im-premium-card",
  ".im-cap-card",
  ".im-use-card",
  ".im-step-card",
  ".fp-card",
  ".fp-step",
  ".fp-panel",
  ".fp-report-card span",
].join(",");

function splitTextElement(element) {
  if (element.dataset.motionPrepared === "true") return;
  const text = element.textContent.trim().replace(/\s+/g, " ");
  if (!text) return;

  element.dataset.motionPrepared = "true";
  element.setAttribute("aria-label", text);
  element.textContent = "";

  text.split(" ").forEach((word, index) => {
    const mask = document.createElement("span");
    mask.className = "nx-word-mask";
    mask.setAttribute("aria-hidden", "true");
    mask.style.setProperty("--word-index", index);

    const inner = document.createElement("span");
    inner.className = "nx-word";
    inner.textContent = word;
    mask.appendChild(inner);
    element.appendChild(mask);

    if (index < text.split(" ").length - 1) {
      element.appendChild(document.createTextNode(" "));
    }
  });
}

export function TypographyMotionProvider() {
  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const root = document.documentElement;

    root.classList.add("nx-typography-motion");

    const masked = Array.from(document.querySelectorAll(maskedSelectors));
    const reveal = Array.from(document.querySelectorAll(revealSelectors));
    const stagger = Array.from(document.querySelectorAll(staggerSelectors));

    masked.forEach((element) => {
      splitTextElement(element);
      element.classList.add("nx-masked-text");
    });

    reveal.forEach((element) => element.classList.add("nx-reveal-text"));
    stagger.forEach((element, index) => {
      element.classList.add("nx-stagger-item");
      element.style.setProperty("--stagger-index", index % 8);
    });

    const animated = [...masked, ...reveal, ...stagger];

    if (reducedMotion) {
      animated.forEach((element) => element.classList.add("is-visible"));
      return () => root.classList.remove("nx-typography-motion");
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.12 }
    );

    animated.forEach((element) => observer.observe(element));

    // No per-frame scroll work at all: every remaining text effect is driven
    // by IntersectionObserver + CSS transitions, so scrolling costs nothing.
    return () => {
      observer.disconnect();
      root.classList.remove("nx-typography-motion");
    };
  }, []);

  return null;
}

export default TypographyMotionProvider;
