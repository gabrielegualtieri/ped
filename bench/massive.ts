/**
 * Accuracy benchmark through the public API, on the MASSIVE scenario test set (Amazon, CC BY 4.0): the
 * same utterances translated into 51 languages, each labelled with one of 18 scenarios.
 *
 * Every utterance is one `Router.systemOne` call with three questions: the scenario (an 18-option
 * choice), and two yes/no questions, "is this request about <its scenario>?" and "... about <another
 * scenario>?". The router picks the checkpoint as in production (no language hint).
 *
 *   npx tsx bench/massive.ts --langs en,it,zh --n 100 [--options '{"precision":"int8"}']
 *                            [--english DIR --multilingual DIR] [--module path/to/index.ts] [--out file.json]
 *
 * `--module` benchmarks another build of the library (e.g. a checkout of an older tag), so versions are
 * compared on byte-identical requests. Only API present since 0.2.0 is used.
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { cpus } from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import { gunzipSync } from "node:zlib";
import { pathToFileURL } from "node:url";

const SCENARIOS: Record<string, string> = {
  alarm: "alarms and wake-up times",
  audio: "volume, mute, audio settings",
  calendar: "events, meetings, reminders",
  cooking: "recipes and cooking",
  datetime: "current date and time, time zones",
  email: "reading or sending emails",
  general: "chit-chat, jokes, repeat, confirmations",
  iot: "smart home devices: lights, plugs, vacuum, coffee machine",
  lists: "to-do and shopping lists",
  music: "music preferences, liking songs, music settings",
  news: "news and headlines",
  play: "play music, radio, podcasts, audiobooks, games",
  qa: "factual questions, definitions, maths, stocks, currency",
  recommendation: "recommend places, events, movies",
  social: "social media posts and complaints to companies",
  takeaway: "order food delivery or takeaway",
  transport: "trains, taxis, traffic, tickets",
  weather: "weather forecast",
};
const LABELS = Object.keys(SCENARIOS);
const FILES: Record<string, string> = { zh: "zh-CN" };

const { values: args } = parseArgs({
  options: {
    langs: { type: "string", default: "en,it,es,fr,de,pt,ru,zh,ja,ar,hi,ko,tr,pl,th,sw" },
    n: { type: "string", default: "100" },
    options: { type: "string", default: "{}" },
    english: { type: "string" },
    multilingual: { type: "string" },
    module: { type: "string", default: path.resolve(import.meta.dirname, "../src/index.ts") },
    out: { type: "string" },
    data: { type: "string", default: path.resolve(import.meta.dirname, ".data") },
  },
});

interface Row {
  id: string;
  label: string;
  text: string;
}

async function load(lang: string): Promise<Map<string, Row>> {
  const file = path.join(args.data, `${lang}.json.gz`);
  if (!existsSync(file)) {
    await mkdir(args.data, { recursive: true });
    const url = `https://huggingface.co/datasets/mteb/amazon_massive_scenario/resolve/main/test/${FILES[lang] ?? lang}.json.gz`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    await writeFile(file, Buffer.from(await res.arrayBuffer()));
  }
  const lines = gunzipSync(await readFile(file))
    .toString("utf8")
    .split("\n")
    .filter(Boolean);
  return new Map(
    lines.map((l) => {
      const r = JSON.parse(l) as Row;
      return [r.id, r];
    }),
  );
}

// seeded, so every version, model and language answers the same utterances
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle<T>(xs: T[], r: () => number): T[] {
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j] as T, a[i] as T];
  }
  return a;
}

function ece(conf: number[], correct: boolean[], bins = 15): number {
  let e = 0;
  for (let b = 0; b < bins; b++) {
    const sel = conf.flatMap((c, i) => (c > b / bins && c <= (b + 1) / bins ? [i] : []));
    if (!sel.length) continue;
    const meanConf = sel.reduce((s, i) => s + (conf[i] ?? 0), 0) / sel.length;
    const acc = sel.filter((i) => correct[i]).length / sel.length;
    e += (sel.length / conf.length) * Math.abs(meanConf - acc);
  }
  return e;
}

function auroc(scores: number[], labels: boolean[]): number {
  const order = scores.map((s, i) => [s, i] as const).sort((a, b) => a[0] - b[0]);
  const ranks = new Array<number>(scores.length);
  order.forEach(([, i], r) => (ranks[i] = r + 1));
  const pos = labels.filter(Boolean).length;
  const neg = labels.length - pos;
  const sum = labels.reduce((s, l, i) => s + (l ? (ranks[i] ?? 0) : 0), 0);
  return (sum - (pos * (pos + 1)) / 2) / (pos * neg);
}

const quantile = (xs: number[], q: number) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(q * xs.length))] ?? 0;

interface Answer {
  choice?: string;
  noul?: number;
  probabilities?: Record<string, number>;
  answer_confidence?: number;
}
interface RouterLike {
  systemOne(state: unknown, questions: object): Promise<{ answers: Record<string, Answer>; routing: { model: string } }>;
  preload(): Promise<void>;
  close(): Promise<void>;
}

const lib = (await import(pathToFileURL(path.resolve(args.module)).href)) as { Router: new (opts: object) => RouterLike };
const version = (JSON.parse(await readFile(path.resolve(path.dirname(args.module), "../package.json"), "utf8")) as { version: string }).version;
const extra = JSON.parse(args.options) as Record<string, unknown>;
const models = Object.fromEntries(
  (
    [
      ["english", args.english],
      ["multilingual", args.multilingual],
    ] as const
  ).flatMap(([k, d]) => (d ? [[k, { modelDir: d }]] : [])),
);
const router = new lib.Router({ ...extra, models });
await router.preload();

const enRows = await load("en");
const ids = shuffle(
  [...enRows.keys()].sort((a, b) => Number(a) - Number(b)),
  rng(42),
).slice(0, Number(args.n));

const perLang: Record<string, object> = {};
const all = { scenario: [] as boolean[], conf: [] as number[], yesNo: [] as boolean[], pTrue: [] as number[], truth: [] as boolean[], ms: [] as number[] };
for (const lang of args.langs.split(",")) {
  const rows = await load(lang);
  const res = { scenario: [] as boolean[], conf: [] as number[], yesNo: [] as boolean[], ms: [] as number[], routes: {} as Record<string, number> };
  for (const id of ids) {
    const row = rows.get(id);
    if (!row) continue;
    const others = LABELS.filter((l) => l !== row.label);
    const neg = others[Math.floor(rng(Number(id) * 7919 + 13)() * others.length)] ?? "";
    const questions = {
      scenario: { type: "choice", instructions: "Which category does this voice assistant request belong to?", criteria: SCENARIOS },
      about: { type: "noul", instructions: `Is this request about ${SCENARIOS[row.label]}?` },
      aboutOther: { type: "noul", instructions: `Is this request about ${SCENARIOS[neg]}?` },
    };
    const t0 = performance.now();
    const r = await router.systemOne(row.text, questions);
    res.ms.push(performance.now() - t0);
    res.routes[r.routing.model] = (res.routes[r.routing.model] ?? 0) + 1;
    const sc = r.answers["scenario"] ?? {};
    res.scenario.push(sc.choice === row.label);
    res.conf.push(sc.answer_confidence ?? Math.max(...Object.values(sc.probabilities ?? { x: 0 })));
    for (const [q, truth] of [
      ["about", true],
      ["aboutOther", false],
    ] as const) {
      const p = r.answers[q]?.noul ?? 0;
      res.yesNo.push(p > 0.5 === truth);
      all.pTrue.push(p);
      all.truth.push(truth);
    }
  }
  const acc = (xs: boolean[]) => xs.filter(Boolean).length / Math.max(1, xs.length);
  perLang[lang] = {
    n: res.scenario.length,
    scenario_accuracy: acc(res.scenario),
    scenario_ece: ece(res.conf, res.scenario),
    yes_no_accuracy: acc(res.yesNo),
    p50_ms: quantile(res.ms, 0.5),
    routes: res.routes,
  };
  all.scenario.push(...res.scenario);
  all.conf.push(...res.conf);
  all.yesNo.push(...res.yesNo);
  all.ms.push(...res.ms);
  process.stderr.write(`${lang}: ${JSON.stringify(perLang[lang])}\n`);
}
await router.close();

const acc = (xs: boolean[]) => xs.filter(Boolean).length / Math.max(1, xs.length);
const summary = {
  version,
  options: extra,
  bundles: { english: args.english ?? "download", multilingual: args.multilingual ?? "download" },
  hardware: { cpu: cpus()[0]?.model ?? "?", cores: cpus().length, node: process.version },
  n_per_language: Number(args.n),
  languages: args.langs.split(","),
  pooled: {
    scenario_accuracy: acc(all.scenario),
    scenario_ece: ece(all.conf, all.scenario),
    yes_no_accuracy: acc(all.yesNo),
    yes_no_auroc: auroc(all.pTrue, all.truth),
    p50_ms: quantile(all.ms, 0.5),
    p95_ms: quantile(all.ms, 0.95),
  },
  per_language: perLang,
};
console.log(JSON.stringify(summary, null, 1));
if (args.out) await writeFile(args.out, JSON.stringify(summary, null, 1));
