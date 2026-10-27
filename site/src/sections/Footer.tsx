"use client";
import { motion, useReducedMotion } from "motion/react";

// The wordmark closes the page, fully on screen, with the credits under it. The model licences (CC BY) require credit on the page, so the
// credits stay, folded behind one small "Credits" button instead of a block of text.
export default function Footer() {
  const reduce = useReducedMotion();
  return (
    <footer className="gutter overflow-hidden border-t border-line pt-[clamp(56px,8vw,104px)] pb-[calc(env(safe-area-inset-bottom,0px)+28px)]">
      <div className="mx-auto grid max-w-[1200px] gap-8">
        <motion.p
          aria-label="Smoke Rescue"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "0px 0px -5% 0px" }}
          transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 60, damping: 20 }}
          className="select-none whitespace-nowrap bg-[linear-gradient(180deg,#f7eedf_35%,#f98c0a_140%)] bg-clip-text py-[0.04em] text-center font-stencil text-[clamp(2.4rem,11.6vw,10rem)] leading-none tracking-[0.01em] text-transparent"
        >
          SMOKE RESCUE
        </motion.p>
        <details className="group justify-self-center text-center text-footnote text-mute">
          <summary className="mx-auto w-fit cursor-pointer list-none rounded-full border border-line px-3 py-1.5 font-semibold transition-colors hover:text-ink [&::-webkit-details-marker]:hidden">
            Credits
          </summary>
          <p className="mt-3 max-w-[60ch] leading-relaxed">
            Robot dog based on &quot;Spot robot low poly model with rig&quot; by CG Material, repainted, branding removed.
            Firefighter by jakobscheidt. Ural4320 fire truck by Ford mondeo. All from Sketchfab, CC BY.
            Textures and lighting from Poly Haven, CC0. Music &quot;Impact Prelude&quot; by Kevin MacLeod, CC BY 4.0.
          </p>
        </details>
      </div>
    </footer>
  );
}
