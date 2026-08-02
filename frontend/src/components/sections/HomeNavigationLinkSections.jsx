import { useRef } from "react";
import { motion, useInView, useReducedMotion } from "framer-motion";
// Transparent-background cutout: the original hand-hd.png has a warm-white
// panel baked in that showed as a beige rectangle on the creme home base.
import handImage from "../../assets/hand-hd-cutout.png";
import "./HomeNavigationLinkSections.css";

/**
 * "Caught by the hand" choreography: once the slide is in view, the title
 * falls from above and lands on an underdamped spring — it dips just past
 * its resting point and eases back up, like something with real weight being
 * caught. The hand rises to meet it, then gives a small dip at the moment of
 * the catch to absorb the landing.
 */
export function HomeNavigationLinkSections() {
  const sectionRef = useRef(null);
  // No `once`: while the slide is off screen everything resets instantly (and
  // invisibly), so the catch replays every time the slide scrolls back in.
  const inView = useInView(sectionRef, { amount: 0.4 });
  const reducedMotion = useReducedMotion();

  const show = reducedMotion || inView;

  return (
    <section className="nx-home-fingerprint-slide" ref={sectionRef}>
      <motion.h2
        initial={reducedMotion ? false : { y: "-46vh", opacity: 0, scale: 1.05 }}
        animate={show ? { y: 0, opacity: 1, scale: 1 } : { y: "-46vh", opacity: 0, scale: 1.05 }}
        transition={
          show
            ? {
                y: { type: "spring", stiffness: 46, damping: 5, mass: 1.4, delay: 0.08 },
                scale: { type: "spring", stiffness: 46, damping: 10, mass: 1.4, delay: 0.08 },
                opacity: { duration: 0.5, ease: "easeOut", delay: 0.08 },
              }
            : { duration: 0 }
        }
      >
        Automated Forensic Examination
      </motion.h2>

      <motion.p
        className="nx-catch-sub"
        initial={reducedMotion ? false : { y: 14, opacity: 0 }}
        animate={show ? { y: 0, opacity: 1 } : { y: 14, opacity: 0 }}
        transition={
          show
            ? { duration: 0.8, ease: [0.16, 1, 0.3, 1], delay: 1.15 }
            : { duration: 0 }
        }
      >
        Every piece of evidence, handled with care.
      </motion.p>

      <div className="nx-catch-hand-wrap" aria-hidden="true">
        <motion.img
          className="nx-catch-hand"
          src={handImage}
          alt=""
          loading="lazy"
          decoding="async"
          initial={reducedMotion ? false : { y: 34, opacity: 0 }}
          animate={
            show
              ? { y: [34, 0, 0, 26, -7, 17, -4, 10, -2, 5, 0], opacity: 1 }
              : { y: 34, opacity: 0 }
          }
          transition={
            show
              ? {
                  y: {
                    duration: 4.4,
                    times: [0, 0.14, 0.23, 0.33, 0.43, 0.53, 0.63, 0.72, 0.81, 0.9, 0.98],
                    ease: "easeInOut",
                  },
                  opacity: { duration: 0.7, ease: "easeOut" },
                }
              : { duration: 0 }
          }
        />
      </div>
    </section>
  );
}

export default HomeNavigationLinkSections;
