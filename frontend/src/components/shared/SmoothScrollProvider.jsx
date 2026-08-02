import { useEffect } from "react";
import Lenis from "lenis";
import "lenis/dist/lenis.css";

/**
 * Lenis-powered inertial scrolling for the marketing site.
 *
 * Lenis drives the real document scroll position, so native scroll events
 * still fire and framer-motion's useScroll/useTransform hooks stay in sync.
 * Touch devices keep fully native scrolling (syncTouch stays off), and
 * reduced-motion users are left with the browser default.
 */
export function SmoothScrollProvider() {
  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reducedMotion) return undefined;

    // lerp 0.12: longer, more liquid coast after each wheel tick — the
    // "butter" glide. Safe now that nothing on the scroll path does per-frame
    // work; the generous multiplier keeps response feeling immediate.
    const lenis = new Lenis({
      autoRaf: true,
      lerp: 0.12,
      wheelMultiplier: 1.3,
      anchors: { offset: -90 },
    });

    return () => lenis.destroy();
  }, []);

  return null;
}

export default SmoothScrollProvider;
