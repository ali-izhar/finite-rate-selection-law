// Decode the QR code from the rendered preview (poster-detail.png) at the positions build.mjs
// recorded (poster.qr.json) and compare with the intended URLs.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { PNG } = require("pngjs");
const jsQR = require("jsqr");
const png = PNG.sync.read(readFileSync(new URL("./poster-detail.png", import.meta.url)));
const ppi = png.height / 36;
const rects = JSON.parse(readFileSync(new URL("./poster.qr.json", import.meta.url), "utf8"));
const want = {
  "assets/qr-repo.svg": "https://github.com/ali-izhar/finite-rate-selection-law",
};
let ok = rects.length === 1;
for (const r of rects) {
  const m = 0.15;
  const x = Math.round((r.x - m) * ppi), y = Math.round((r.y - m) * ppi), w = Math.round((r.w + 2 * m) * ppi), h = w;
  const buf = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
    const s = ((y + i) * png.width + (x + j)) * 4, d = (i * w + j) * 4;
    for (let k = 0; k < 4; k++) buf[d + k] = png.data[s + k];
  }
  const url = want[r.src], got = jsQR(buf, w, h)?.data;
  console.log(got === url ? "QR ok  " : "QR FAIL", url, got === url ? "" : `(got ${got})`);
  ok &&= got === url;
}
process.exit(ok ? 0 : 1);
