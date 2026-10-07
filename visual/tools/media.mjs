// Copy the narrated film's web files into visual/media/ (committed) and write
// media/media.json, the manifest the page reads. Rerunnable; needs ffmpeg + ffprobe.
//
// Usage (from visual/):  node tools/media.mjs
//
// Source: ../film/build/web/ (made by `./build.sh assemble` in visual/film/), plus
// ../film/build/film.offsets.json (scene starts in the film) and
// ../film/scenes.json (scene → page chapter).
// Output:
//   media/film-720p.mp4 + .vtt.txt  the whole narrated film (one file; captions as .vtt.txt
//                                        because the host does not serve text/vtt)
//   media/film-poster.jpg           its poster (the hook's title card, 1280x720)
//   media/<scene>.mp4 + .vtt.txt + .jpg      one narrated clip per chapter; the poster is the
//                                        frame at the scene's authored "poster" time in
//                                        scenes.json (a settled frame after the last cue and
//                                        before any fade-out, chosen by eye)
//   media/media.json                     { film: {src, vtt, poster, duration, chapters[], play?},
//                                          clips: {<chapter id>: {scene, src, vtt, poster, duration, play?}} }
// A scene may give "play": [x, y], the centre of the page's play mark in its poster, in pixels
// of the 1280x720 frame (an empty patch, chosen by eye); without it the mark sits lower left.
// media.json carries it as percentages of the frame: play: {x, y}. The first scene may also give
// "film_play", the same for the film's poster (the hook's title card): media.json film.play.
// Every file must stay under 15 MB and the folder under 60 MB (the publisher's limits);
// the script exits 1 otherwise.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FILM = path.resolve(ROOT, 'film');
const WEB = path.join(FILM, 'build', 'web');
const OUT = path.join(ROOT, 'media');
const FILE_MAX = 15e6, TOTAL_MAX = 60e6;

function run(cmd, args) {
  const r = spawnSync(cmd, args, { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(cmd + ' failed: ' + (r.stderr || '').trim().split('\n').pop());
  return r.stdout;
}
const duration = (f) => +(+run('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', f]).trim()).toFixed(3);
function poster(src, t, dst) {
  run('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-ss', t.toFixed(3), '-i', src, '-frames:v', '1',
    '-vf', 'scale=1280:720:flags=lanczos', '-q:v', '3', dst]);
}

const filmSrc = path.join(WEB, 'film-720p.mp4');
if (!fs.existsSync(filmSrc)) {
  console.error('missing ' + path.relative(ROOT, filmSrc) + ': run ./build.sh assemble in visual/film/ first');
  process.exit(1);
}
const offs = JSON.parse(fs.readFileSync(path.join(FILM, 'build', 'film.offsets.json'), 'utf8'));
const scenes = JSON.parse(fs.readFileSync(path.join(FILM, 'scenes.json'), 'utf8'));

fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });
const cp = (f, to = f) => fs.copyFileSync(path.join(WEB, f), path.join(OUT, to));

cp('film-720p.mp4'); cp('film-720p.vtt', 'film-720p.vtt.txt');
poster(path.join(FILM, 'build', 'film-poster.png'), 0, path.join(OUT, 'film-poster.jpg'));
const film = {
  src: 'media/film-720p.mp4', vtt: 'media/film-720p.vtt.txt', poster: 'media/film-poster.jpg',
  duration: duration(path.join(OUT, 'film-720p.mp4')),
  chapters: offs.scenes.map((o) => {
    const s = scenes.find((x) => x.id === o.id);
    return { scene: o.id, chapter: s.chapter, title: s.title, start: +o.start.toFixed(3) };
  }),
};
// a placed play mark: [x, y] in pixels of the 1280x720 frame -> {x, y} in percent, or exit 1
function placed(v, where) {
  const [x, y] = Array.isArray(v) ? v : [];
  if (!(Array.isArray(v) && v.length === 2 && x >= 0 && x <= 1280 && y >= 0 && y <= 720)) {
    console.error(`scenes.json: ${where} is bad: give [x, y] in pixels of the 1280x720 frame`);
    process.exit(1);
  }
  return { x: +(x / 12.8).toFixed(2), y: +(y / 7.2).toFixed(2) };
}
// the film's poster is the hook's title card; the first scene may place the film's mark ("film_play")
if (scenes[0].film_play !== undefined) film.play = placed(scenes[0].film_play, `scene "${scenes[0].id}" "film_play"`);
const clips = {};
for (const s of scenes) {
  cp(s.id + '.mp4'); cp(s.id + '.vtt', s.id + '.vtt.txt');
  const f = path.join(OUT, s.id + '.mp4'), d = duration(f);
  if (typeof s.poster !== 'number' || s.poster < 0 || s.poster > d - 0.1) {
    console.error(`scenes.json: scene "${s.id}" needs a "poster" time in [0, ${(d - 0.1).toFixed(2)}] s (a settled frame, chosen by eye)`);
    process.exit(1);
  }
  poster(f, s.poster, path.join(OUT, s.id + '.jpg'));
  clips[s.chapter] = { scene: s.id, title: s.title, src: 'media/' + s.id + '.mp4', vtt: 'media/' + s.id + '.vtt.txt', poster: 'media/' + s.id + '.jpg', duration: d };
  if (s.play !== undefined) clips[s.chapter].play = placed(s.play, `scene "${s.id}" "play"`);
}
fs.writeFileSync(path.join(OUT, 'media.json'), JSON.stringify({ film, clips }, null, 1) + '\n');

let total = 0, bad = 0;
for (const f of fs.readdirSync(OUT).sort()) {
  const b = fs.statSync(path.join(OUT, f)).size;
  total += b;
  if (b >= FILE_MAX) { bad++; console.error(`OVER 15 MB: ${f} ${(b / 1e6).toFixed(2)} MB`); }
}
console.log(`media/: ${fs.readdirSync(OUT).length} files, ${(total / 1e6).toFixed(1)} MB; film ${film.duration.toFixed(1)} s`);
if (total >= TOTAL_MAX) { bad++; console.error('OVER 60 MB in total'); }
process.exit(bad ? 1 : 0);
