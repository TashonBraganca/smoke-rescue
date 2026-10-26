"use client";
import { useEffect, useState } from "react";

const LINKS = [["#mission", "Watch a mission"], ["#try", "Test a photo"], ["#house", "The house"], ["#slam", "SLAM"], ["#results", "Results"], ["#built", "How it's built"]];

// Floating pill; slides away while scrolling down so it never sits on content, returns on scroll up.
export default function Nav({ rescue = false }: { rescue?: boolean }) {
  const [away, setAway] = useState(false);
  useEffect(() => {
    let last = window.scrollY;
    const on = () => { const y = window.scrollY; setAway(y > last && y > 120); last = y; };
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);
  return (
    <nav
      aria-label="Main"
      className={`fixed left-1/2 top-[calc(env(safe-area-inset-top,0px)+14px)] z-50 flex max-w-[calc(100vw-24px)] -translate-x-1/2 items-center gap-5 rounded-full border border-white/10 bg-stage/92 py-2 pl-5 pr-2 transition-all duration-500 ease-[var(--ease-spring)] ${away ? "-translate-y-[140%] opacity-0" : ""}`}
    >
      <a href={rescue ? "/#top" : "#top"} className="whitespace-nowrap font-stencil text-lg tracking-wide text-ink">SMOKE RESCUE</a>
      <div className="hidden gap-5 text-sm font-semibold text-sub lg:flex">
        {LINKS.map(([h, l]) => <a key={h} href={rescue ? `/${h}` : h} className="whitespace-nowrap transition-colors hover:text-ink">{l}</a>)}
      </div>
      <a href="/rescue/" className="whitespace-nowrap rounded-full bg-heat px-4 py-2.5 text-sm font-extrabold text-heat-ink transition-colors hover:bg-amber">Try it</a>
    </nav>
  );
}
