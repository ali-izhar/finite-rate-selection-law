// Build the poster: QR codes -> inline figures + KaTeX -> poster.render.html -> poster.pdf, poster.png
// Usage: node poster/build.mjs   (run make_figures.py first; build.sh does both)
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { homedir } from "node:os";

const HERE = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const katex = require("katex");
const QRCode = require("qrcode");
const { chromium } = require(join(HERE, "..", "visual", "node_modules", "playwright"));

const CHROME = process.env.CHROME_PATH || join(homedir(),
  ".cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux-arm64/chrome-headless-shell");

// ---------------------------------------------------------------- 1. QR code (vector, ECC M, 2-module quiet zone inside the SVG; the header keeps >= 0.3 in of empty ground around it)
const QRS = {
  "qr-repo.svg": "https://github.com/ali-izhar/finite-rate-selection-law",
};
for (const [file, url] of Object.entries(QRS)) {
  const svg = await QRCode.toString(url, {
    type: "svg", errorCorrectionLevel: "M", margin: 2,
    color: { dark: "#141518ff", light: "#F7F6F2ff" },
  });
  writeFileSync(join(HERE, "assets", file), svg);
}

// ---------------------------------------------------------------- 2. inline figures, render math
let html = readFileSync(join(HERE, "poster.html"), "utf8");
for (const f of readdirSync(join(HERE, "figures")).filter((f) => /^V[A-Za-z0-9]+\.svg$/.test(f))) {
  const svg = readFileSync(join(HERE, "figures", f), "utf8");
  html = html.replace(`<!--#${f.replace(".svg", "")}-->`, svg);
}
const missing = html.match(/<!--#V[A-Za-z0-9]+-->/g);
if (missing) throw new Error("figures not inlined: " + missing.join(", "));

// \( ... \) -> KaTeX HTML. Only inside <body>, outside SVG.
const [head, body] = html.split("<body>");
const rendered = body.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => {
  const t = tex.replace(/&gt;/g, ">").replace(/&lt;/g, "<").replace(/&amp;/g, "&");
  return katex.renderToString(t, { throwOnError: true, output: "html", strict: "ignore" });
});
html = head + "<body>" + rendered;
const OUT_HTML = join(HERE, "poster.render.html");
writeFileSync(OUT_HTML, html);

// ---------------------------------------------------------------- 3. render, check, print
const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({ viewport: { width: 2304, height: 3456 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(OUT_HTML).href, { waitUntil: "load" });
await page.evaluate(async () => { await document.fonts.ready; });

const report = await page.evaluate(() => {
  const PX = 96, PT = 96 / 72;
  const out = { overflow: [], small: [], fontsFailed: [] };
  for (const ff of document.fonts) if (ff.status !== "loaded" && ff.status !== "unloaded") out.fontsFailed.push(ff.family);
  // containment: every container's descendants must stay inside its content box
  const conts = ["#header", "#hero .inner", "#hero .frame", "#sec1", "#sec2", "#p1", "#p2", "#p3", "#p4", "#p5", "#p6", "#footer"];
  for (const sel of conts) {
    const el = document.querySelector(sel);
    const cs = getComputedStyle(el), b = el.getBoundingClientRect();
    const box = { l: b.left + parseFloat(cs.paddingLeft), r: b.right - parseFloat(cs.paddingRight),
                  t: b.top + parseFloat(cs.paddingTop), b: b.bottom - parseFloat(cs.paddingBottom) };
    let worst = { r: -1e9, b: -1e9, l: -1e9, t: -1e9 };
    for (const d of el.querySelectorAll("*")) {
      if (d.closest(".katex") && !d.classList.contains("katex")) continue;
      const r = d.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      worst.r = Math.max(worst.r, r.right - box.r); worst.b = Math.max(worst.b, r.bottom - box.b);
      worst.l = Math.max(worst.l, box.l - r.left); worst.t = Math.max(worst.t, box.t - r.top);
    }
    const over = Math.max(worst.r, worst.b, worst.l, worst.t) / PX;
    out.overflow.push({ sel, slack_in: +(-over).toFixed(2), left_slack: +(-worst.l / PX).toFixed(2), top_slack: +(-worst.t / PX).toFixed(2),
      bottom_slack: +(-worst.b / PX).toFixed(2), right_slack: +(-worst.r / PX).toFixed(2) });
  }
  // figure slots: how much room each .vis has around its svg
  out.slots = [...document.querySelectorAll(".card")].map((c) => {
    const f = c.querySelector(".frame"), r = f.getBoundingClientRect();
    const kids = [...f.querySelectorAll("*")].map((d) => d.getBoundingClientRect()).filter((q) => q.width && q.height);
    const ink = kids.length ? { l: Math.min(...kids.map((q) => q.left)), r: Math.max(...kids.map((q) => q.right)),
      t: Math.min(...kids.map((q) => q.top)), b: Math.max(...kids.map((q) => q.bottom)) } : null;
    const tx = c.querySelector(".txt").getBoundingClientRect(), cb = c.getBoundingClientRect();
    return { id: c.id, frame_x: +(r.left / PX).toFixed(3), frame_y: +(r.top / PX).toFixed(3),
      frame_w: +(r.width / PX).toFixed(3), frame_h: +(r.height / PX).toFixed(3),
      ink_w: ink ? +((ink.r - ink.l) / PX).toFixed(2) : 0, ink_h: ink ? +((ink.b - ink.t) / PX).toFixed(2) : 0,
      ink_outside: ink ? +(Math.max(r.left - ink.l, ink.r - r.right, r.top - ink.t, ink.b - r.bottom) / PX).toFixed(3) : 0,
      text_top: +(tx.top / PX).toFixed(3), text_room_below: +((cb.bottom - tx.bottom) / PX).toFixed(2) };
  });
  // smallest text: HTML (outside KaTeX scripts) and SVG
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const n = walker.currentNode;
    if (!n.textContent.trim()) continue;
    const el = n.parentElement;
    let fs = parseFloat(getComputedStyle(el).fontSize) / PT;
    const svg = el.closest("svg");
    if (svg && svg.viewBox.baseVal && svg.viewBox.baseVal.width) {
      // SVG user unit = 1 pt: scale the CSS px size by the rendered/viewBox ratio
      fs = fs * (svg.getBoundingClientRect().width / svg.viewBox.baseVal.width);
    }
    if (fs < 15.95) {
      const inKatex = !!el.closest(".katex");
      out.small.push({ text: n.textContent.trim().slice(0, 30), pt: +fs.toFixed(1), katexScript: inKatex });
    }
  }
  // word count: visible text outside KaTeX (HTML and SVG), plus the number of KaTeX expressions
  const countIn = (sels) => {
    let words = 0;
    for (const sel of sels) for (const root of document.querySelectorAll(sel)) {
      const w2 = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      let buf = "";
      while (w2.nextNode()) {
        const n = w2.currentNode, el = n.parentElement;
        if (el.closest(".katex") || el.closest("title") || el.closest("style") || el.closest(".halo")) { buf += " "; continue; }
        buf += (el.closest("svg") || el.closest("td, th") ? " " : "") + n.textContent;
      }
      const toks = buf.split(/[\s|]+/).filter((t) => t && /[\p{L}\p{N}%]/u.test(t));
      words += toks.filter((t) => /\p{L}/u.test(t)).length;
      nums += toks.filter((t) => !/\p{L}/u.test(t)).length;
    }
    return [words, nums];
  };
  let nums = 0;
  out.words = countIn(["#hero", ".sec", ".card"]); out.nums = nums; nums = 0;
  [out.footerWords] = countIn(["#header .qr .t"]);
  out.math = document.querySelectorAll(".katex").length;
  out.qr = [...document.querySelectorAll(".qr img")].map((i) => { const r = i.getBoundingClientRect();
    return { x: r.left / PX, y: r.top / PX, w: r.width / PX, src: i.getAttribute("src") }; });
  return out;
});
console.log(`prose words (tokens with a letter, outside math): panels ${report.words[0]} + QR label ${report.footerWords} = ${report.words[0] + report.footerWords}; number-only tokens ${report.words[1]}; math expressions: ${report.math}`);
for (const sl of report.slots) console.log("  slot", JSON.stringify(sl));
const real = report.small.filter((s) => !s.katexScript);
console.log("block slack (in; negative = overflow; ok* = line-box leading under 0.06 in, no ink outside):");
for (const o of report.overflow) console.log(`  ${o.slack_in >= 0 ? " ok " : o.slack_in >= -0.06 ? " ok*" : "OVER"} ${o.sel.padEnd(14)} min slack ${o.slack_in} in; bottom ${o.bottom_slack}; right ${o.right_slack}; left ${o.left_slack}; top ${o.top_slack}`);
console.log(`text below 16 pt: ${real.length} (KaTeX sub/superscripts below 16 pt: ${report.small.length - real.length})`);
for (const s of real.slice(0, 20)) console.log("   ", s);
if (report.fontsFailed.length) console.log("fonts not loaded:", report.fontsFailed);
const fams = await page.evaluate(() => [...document.fonts].filter((f) => f.status === "loaded").map((f) => f.family + " " + f.style));
console.log("fonts loaded:", [...new Set(fams)].join(", "));

writeFileSync(join(HERE, "poster.qr.json"), JSON.stringify(report.qr));
await page.pdf({ path: join(HERE, "poster.pdf"), width: "24in", height: "36in", printBackground: true, preferCSSPageSize: true });
// preview ~2400 px tall
await page.setViewportSize({ width: 2304, height: 3456 });
const scale = 2400 / 3456;
const page2 = await browser.newPage({ viewport: { width: 2304, height: 3456 }, deviceScaleFactor: scale });
await page2.goto(pathToFileURL(OUT_HTML).href, { waitUntil: "load" });
await page2.evaluate(async () => { await document.fonts.ready; });
await page2.screenshot({ path: join(HERE, "poster.png"), fullPage: false });
const page3 = await browser.newPage({ viewport: { width: 2304, height: 3456 }, deviceScaleFactor: 1.5 });
await page3.goto(pathToFileURL(OUT_HTML).href, { waitUntil: "load" });
await page3.evaluate(async () => { await document.fonts.ready; });
await page3.screenshot({ path: join(HERE, "poster-detail.png"), fullPage: false });
await browser.close();
console.log("wrote poster.pdf, poster.png, poster-detail.png");
