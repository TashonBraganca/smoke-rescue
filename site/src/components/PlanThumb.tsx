import { KIND_COLOR, type Plan } from "@/lib/plans";

// A small floor plan in SVG: rooms by kind, the fire as a glow, the front door as a bright dot. Units are metres.
export default function PlanThumb({ plan, id, className = "" }: { plan: Plan; id: string; className?: string }) {
  const [fx0, fz0, fx1, fz1] = plan.footprint, w = fx1 - fx0, d = fz1 - fz0, pad = 0.6;
  const ids = new Set(plan.rooms.map((r) => r.id));
  const Y = (z: number) => fz1 - z; // north up
  const door = plan.doors.find((x) => x.between.some((b) => !ids.has(b)));
  const r = Math.max(w, d) * 0.012;
  return (
    <svg viewBox={`${fx0 - pad} ${-pad} ${w + pad * 2} ${d + pad * 2}`} className={className} aria-hidden="true">
      <defs>
        <radialGradient id={`fire-${id}`}>
          <stop offset="0" stopColor="#fcffa4" stopOpacity="0.95" />
          <stop offset="0.3" stopColor="#f98c0a" stopOpacity="0.75" />
          <stop offset="1" stopColor="#bc3754" stopOpacity="0" />
        </radialGradient>
      </defs>
      {plan.rooms.map((rm) => {
        const [x0, z0, x1, z1] = rm.rect;
        return <rect key={rm.id} x={x0} y={Y(z1)} width={x1 - x0} height={z1 - z0} fill={KIND_COLOR[rm.kind] ?? "#2b2433"} stroke="rgba(247,238,223,.6)" strokeWidth="1.25" vectorEffect="non-scaling-stroke" />;
      })}
      {plan.fires.map((f, i) => <circle key={i} cx={f.at[0]} cy={Y(f.at[1])} r={Math.max(w, d) * 0.16} fill={`url(#fire-${id})`} />)}
      {door && <circle cx={door.at[0]} cy={Y(door.at[1])} r={r * 2.4} fill="#fcffa4" />}
    </svg>
  );
}
