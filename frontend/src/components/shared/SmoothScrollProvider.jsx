import { useEffect } from "react";

export function SmoothScrollProvider() {
  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const finePointer = window.matchMedia("(pointer: fine)").matches;

    if (reducedMotion || !finePointer) return undefined;

    let target = window.scrollY;
    let current = window.scrollY;
    let frame = 0;

    const stop = () => {
      if (frame) window.cancelAnimationFrame(frame);
      frame = 0;
    };

    const tick = () => {
      const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
      target = Math.max(0, Math.min(target, maxScroll));
      current += (target - current) * 0.16;

      if (Math.abs(target - current) < 0.35) {
        current = target;
        window.scrollTo(0, current);
        frame = 0;
        return;
      }

      window.scrollTo(0, current);
      frame = window.requestAnimationFrame(tick);
    };

    const onWheel = (event) => {
      if (
        event.ctrlKey ||
        event.metaKey ||
        event.defaultPrevented ||
        event.target.closest("textarea, input, select, [contenteditable], [data-native-scroll]")
      ) {
        return;
      }

      event.preventDefault();
      target += event.deltaY * 0.92;

      if (!frame) {
        current = window.scrollY;
        frame = window.requestAnimationFrame(tick);
      }
    };

    const onKeyDown = (event) => {
      if (["Home", "End", "PageUp", "PageDown", " ", "ArrowUp", "ArrowDown"].includes(event.key)) {
        stop();
        target = window.scrollY;
        current = target;
      }
    };

    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("keydown", onKeyDown);

    return () => {
      stop();
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  return null;
}

export default SmoothScrollProvider;
