"use client";
import { useRef } from "react";
import { motion, useInView, useReducedMotion } from "motion/react";
import RiseHead from "@/components/RiseHead";

// Results in plain words. The headline numbers are the 120-run headless benchmark, totalled over every run
// (smoke-rescue/data/evidence/bench_25oct_score.json and .txt, scored 25 Oct 2026 by tools/bench/score_bench.py).

function Card({ i, big, small, note }: { i: number; big: string; small: string; note?: string }) {
  const ref = useRef<HTMLDivElement>(null), reduce = useReducedMotion();
  const inView = useInView(ref, { once: true, margin: "0px 0px -10% 0px" });
  return (
    <motion.div ref={ref} initial={reduce ? false : { opacity: 0, y: 24 }} animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ type: "spring", stiffness: 80, damping: 20, delay: i * 0.06 }}
      className="grid content-start gap-1 rounded-3xl border border-line bg-panel/60 p-5">
      <span className="font-stencil text-title2 tabular-nums text-whiteheat">{big}</span>
      <span className="text-callout text-ink">{small}</span>
      {note && <span className="text-footnote text-mute">{note}</span>}
    </motion.div>
  );
}

export default function Results() {
  return (
    <section id="results" className="gutter py-[clamp(80px,11vw,140px)]" aria-label="Results">
      <div className="mx-auto grid max-w-[1200px] gap-14">
        <div className="grid gap-6">
          <RiseHead eyebrow="Results" title="120 runs. 30 houses." />
          <p className="max-w-[62ch] text-body text-sub">
            Every run is scored against the true house, which the robots never get to read. The runs use our headless simulator and
            the same robot brain, with the radio on and off, and with one robot or two.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Card i={0} big="580 of 600" small="people found (97%)" />
          <Card i={1} big="96%" small="of rooms entered" note="997 of 1,036" />
          <Card i={2} big="95%" small="of the floor searched, median run" />
          <Card i={3} big="715 of 721" small="confirmations correct (99%)" />
        </div>

        <figure className="grid gap-3">
          <video src="/evidence/run_family_ranch_5of5.mp4" poster="/evidence/run_family_ranch_poster.jpg" controls muted playsInline preload="none"
            className="w-full rounded-3xl border border-line bg-black" />
          <figcaption className="max-w-[72ch] text-callout text-sub">
            One full live run in our Unity simulator, shown at 3× speed. The robots found all 5 people with no wrong
            confirmations, then held in the hallway when the fire closed the only exit. They never drive through flames.
          </figcaption>
        </figure>

      </div>
    </section>
  );
}
