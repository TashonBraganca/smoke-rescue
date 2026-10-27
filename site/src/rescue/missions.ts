import type { House } from './data';
export type Film = { src: string | null; peopleFound: number | null; peopleTotal: number; missionSeconds: number | null; allOut: boolean | null; note: string; judge?: { src: string; runStart: number; slow: number } };
export type MissionConfig = { films: Record<string, Film>; intro: string; outro: string; poster: string; engine: string | null };
export const isFilmHouse = (house: House) => house.source === 'ours';
export const missionHref = (house: House, robots: number) => `/rescue/?house=${encodeURIComponent(house.id)}&robots=${isFilmHouse(house) ? 2 : robots}&start=1`;
export function fireRoom(house: House) {
  const point = house.fire_origins?.[0];
  if (!point) return 'Fire inside';
  const room = house.rooms_poly.find(r => inside(r.poly, point[0], point[1]));
  return room ? `Fire in the ${room.name.toLowerCase().replaceAll('_', ' ')}` : 'Fire inside';
}
function inside(poly: number[][], x: number, y: number) {
  let yes = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [a, b] = poly[i], [c, d] = poly[j];
    if ((b > y) !== (d > y) && x < (c - a) * (y - b) / (d - b) + a) yes = !yes;
  }
  return yes;
}
