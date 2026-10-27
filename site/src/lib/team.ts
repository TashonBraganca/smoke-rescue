import { SKIN_NAMES, type SkinName } from "./skins";

/** The dogs in a team of n, in order: Cinder, then Ember, then Ash. */
export const teamOf = (n: number): SkinName[] => SKIN_NAMES.slice(0, n);

/** "Cinder", "Cinder and Ember", "Cinder, Ember and Ash". */
export function teamPhrase(n: number) {
  const t = teamOf(n);
  return t.length < 2 ? t.join("") : `${t.slice(0, -1).join(", ")} and ${t[t.length - 1]}`;
}

export const dogsWord = (n: number) => (n === 1 ? "1 dog" : `${n} dogs`);
