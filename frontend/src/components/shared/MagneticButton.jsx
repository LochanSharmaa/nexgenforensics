import { useState } from "react";
import { motion } from "framer-motion";
import { cx } from "../../utils/cx";
import "./MagneticButton.css";

/**
 * A button component that tracks the mouse and pulls toward it using spring physics,
 * providing a tactile "magnetic" feel on hover.
 * 
 * @param {Object} props
 * @param {React.ReactNode} props.children - Inner content of the button.
 * @param {"primary"|"secondary"} [props.variant="primary"] - Visual style variation.
 */
export function MagneticButton({ children, variant = "primary" }) {
  const [pos, setPos] = useState({ x: 0, y: 0 });

  const handleMouseMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setPos({
      x: event.clientX - rect.left - rect.width / 2,
      y: event.clientY - rect.top - rect.height / 2,
    });
  };

  const handleMouseLeave = () => {
    setPos({ x: 0, y: 0 });
  };

  return (
    <motion.button
      type="button"
      className={cx("nx-btn", variant === "secondary" && "nx-btn-secondary")}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      animate={{ x: pos.x * 0.07, y: pos.y * 0.12 }}
      transition={{ type: "spring", stiffness: 220, damping: 18 }}
    >
      {children}
    </motion.button>
  );
}

export default MagneticButton;
