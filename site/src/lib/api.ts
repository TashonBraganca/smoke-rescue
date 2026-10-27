// Our live vision API on AWS (Lambda Function URL in front of the agent service). CORS allows this site's origin.
export const API = "https://ajfjm7evl33gu6ptprty6hfthq0ukygx.lambda-url.us-east-1.on.aws";

export type Visibility = { laplacian_var: number; rms_contrast: number; dark_channel: number; opencv: string; ms: number };

// The robot's own camera inside the house in clear air, measured by the same endpoint (deploy/aws/samples/sample_rgb.png).
export const CLEAR_REF: Visibility = { laplacian_var: 338.9, rms_contrast: 53.4, dark_channel: 0.214, opencv: "5.0.0", ms: 6.2 };

const MAX_SIDE = 1280, MAX_BYTES = 4 * 1024 * 1024;

// Downscale in the browser to at most 1280 px on the long side, as JPEG, so uploads stay small and fast.
export async function shrink(src: Blob): Promise<Blob> {
  const bmp = await createImageBitmap(src);
  const k = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
  c.getContext("2d")!.drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close();
  const out = await new Promise<Blob | null>((ok) => c.toBlob(ok, "image/jpeg", 0.9));
  if (!out) throw new Error("This browser could not read that image.");
  if (out.size > MAX_BYTES) throw new Error("That photo is still over 4 MB after shrinking. Try a smaller one.");
  return out;
}

export async function assessVisibility(img: Blob): Promise<{ v: Visibility; roundTripMs: number }> {
  const fd = new FormData();
  fd.append("image", img, "photo.jpg");
  const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 30000), t0 = performance.now();
  try {
    const r = await fetch(`${API}/api/vision/assess_visibility`, { method: "POST", body: fd, signal: ctl.signal });
    if (!r.ok) throw new Error(`The server answered ${r.status}. Try again in a moment.`);
    const v = (await r.json()) as Visibility;
    if (typeof v.laplacian_var !== "number") throw new Error("The server answered, but without measurements.");
    return { v, roundTripMs: performance.now() - t0 };
  } catch (e) {
    if ((e as Error).name === "AbortError") throw new Error("The server took more than 30 seconds. Try again.");
    throw e instanceof TypeError ? new Error("Could not reach our server. Check your connection and try again.") : e;
  } finally { clearTimeout(timer); }
}
