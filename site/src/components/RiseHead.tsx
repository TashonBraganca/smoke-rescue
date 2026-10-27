"use client";
import { useRef } from "react";
import { useInView } from "motion/react";
import WordRise from "@/components/WordRise";

// Section header: small eyebrow plus a WordRise headline. The in-view check sits on the wrapper because
// WordRise's own whileInView watches words that start translated out of their clipping mask, so the
// observer never sees them; driving `show` from here makes the rise fire reliably.
export default function RiseHead({ eyebrow, title, className = "", titleClass = "" }: {
  eyebrow: string; title: string; className?: string; titleClass?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -12% 0px" });
  return (
    <div ref={ref} className={`grid gap-4 ${className}`}>
      <span className="text-footnote font-extrabold uppercase tracking-[0.18em] text-heat">{eyebrow}</span>
      <WordRise
        text={title}
        show={inView}
        className={`text-title1 font-extrabold text-ink [text-wrap:balance] ${titleClass}`}
      />
    </div>
  );
}
