// Static server for visual/ on a fixed port (default 8765), for manual viewing.
// Usage: node tools/serve.mjs [port]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.mp4': 'video/mp4', '.vtt': 'text/vtt',
  '.woff2': 'font/woff2' };
// byte ranges, so a <video> can seek (browsers ask for media in ranges)
function sendFile(req, res, p, type) {
  const size = fs.statSync(p).size, m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || '');
  if (!m || (m[1] === '' && m[2] === '')) {
    res.writeHead(200, { 'Content-Type': type, 'Content-Length': size, 'Accept-Ranges': 'bytes' });
    return fs.createReadStream(p).pipe(res);
  }
  let a = m[1] === '' ? Math.max(0, size - +m[2]) : +m[1], b = m[1] !== '' && m[2] !== '' ? Math.min(+m[2], size - 1) : size - 1;
  if (a > b || a >= size) { res.writeHead(416, { 'Content-Range': `bytes */${size}` }); return res.end(); }
  res.writeHead(206, { 'Content-Type': type, 'Content-Length': b - a + 1, 'Content-Range': `bytes ${a}-${b}/${size}`, 'Accept-Ranges': 'bytes' });
  fs.createReadStream(p, { start: a, end: b }).pipe(res);
}
http.createServer((req, res) => {
  let p = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
  if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, 'index.html');
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { res.writeHead(404); return res.end(); }
  sendFile(req, res, p, TYPES[path.extname(p)] || 'application/octet-stream');
}).listen(+(process.argv[2] || 8765), () => console.log('serving visual/ on http://127.0.0.1:' + (process.argv[2] || 8765)));
