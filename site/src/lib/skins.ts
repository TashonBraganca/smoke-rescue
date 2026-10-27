// Dog paint jobs. Hex is the swatch colour. `shade` scales the 3D body colour: the scenes light the shell two to
// three times harder than the dark parts, so bright paints are stored darker or they bleach to peach under tone
// mapping; the grey keeps more of its value or it renders as black.
export const SKINS = {
  Cinder: { hex: "#E4552A", shade: 0.5, legs: 0x2b2e34, line: "Orange. The one the crew spots first." },
  Ember: { hex: "#D8861A", shade: 0.5, legs: 0x2b2e34, line: "Amber, for the thickest smoke." },
  Ash: { hex: "#8A8F96", shade: 0.85, legs: 0x24272c, line: "Graphite. Quiet on camera." },
} as const;
export type SkinName = keyof typeof SKINS;
export const SKIN_NAMES = Object.keys(SKINS) as SkinName[];
