"use client";
import { motion, useInView, useReducedMotion } from "motion/react";
import { useRef, type ElementType } from "react";

// Headline words slide up from behind a mask, one after another, on a spring.
// `show` lets a scroll-driven scene control it; otherwise it plays once when the headline enters the viewport.
// The in-view check watches the whole headline element: the words themselves start hidden inside their masks,
// so watching them would never fire.
export default function WordRise({
  text, as = "h2", className = "", delay = 0, stagger = 0.06, show,
}: { text: string; as?: ElementType; className?: string; delay?: number; stagger?: number; show?: boolean }) {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -10% 0px" });
  const visible = show !== undefined ? show : (reduce || inView);
  const Tag = as;
  const words = text.split(" ");
  return (
    <Tag ref={ref} className={className}>
      <span className="sr-only">{text}</span>
      {words.map((w, i) => (
        <span key={i} aria-hidden="true" className="inline-block overflow-hidden pb-[0.08em] -mb-[0.08em] align-bottom">
          <motion.span
            className="inline-block will-change-transform"
            initial={{ y: "110%", opacity: 0 }}
            animate={visible ? { y: "0%", opacity: 1 } : { y: "110%", opacity: 0 }}
            transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 140, damping: 22, mass: 0.9, delay: visible ? delay + i * stagger : 0 }}
          >
            {w}
          </motion.span>
          {i < words.length - 1 ? " " : null}
        </span>
      ))}
    </Tag>
  );
}
