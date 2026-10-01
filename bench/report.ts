/**
 * The README's benchmark tables, from the result files of bench/massive.ts and bench/speed.ts.
 *
 *   npx tsx bench/report.ts [--dir bench/results]
 *
 * Every massive-*.json becomes a row of the accuracy tables and every speed-*.json a column of the speed
 * table of its checkpoint, labelled with the library version and the bundle precision they recorded.
 */
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({ options: { dir: { type: "string", default: path.resolve(import.meta.dirname, "results") } } });

interface Hardware {
  cpu: string;
  cores: number;
  node: string;
}
interface Run {
  version: string;
  options: { precision?: string };
  hardware: Hardware;
}
interface Scores {
  scenario_accuracy: number;
  scenario_ece: number;
  yes_no_accuracy: number;
  p50_ms: number;
}
interface Massive extends Run {
  n_per_language: number;
  languages: string[];
  pooled: Scores & { yes_no_auroc: number; p95_ms: number };
  per_language: Record<string, Scores>;
}
interface Speed extends Run {
  dir?: string;
  load_ms: number;
  rss_mb: number;
  cases: { case: string; p50_ms: number }[];
}

const label = (r: Run) => `${r.version}${r.options.precision === "int8" ? " int8" : ""}`;
const byLabel = (a: Run, b: Run) => label(a).localeCompare(label(b), "en", { numeric: true });
const pct = (x: number | undefined) => (x === undefined ? "–" : `${(100 * x).toFixed(1)}%`);
const msec = (x: number | undefined) => (x === undefined ? "–" : `${Math.round(x).toLocaleString("en")} ms`);
const row = (cells: string[]) => `| ${cells.join(" | ")} |`;
const table = (head: string[], rows: string[][]) => [row(head), row(head.map(() => "---")), ...rows.map(row)].join("\n");

async function load<T extends Run>(prefix: string): Promise<T[]> {
  const files = (await readdir(args.dir)).filter((f) => f.startsWith(prefix) && f.endsWith(".json") && !f.endsWith(".tmp.json"));
  const runs = await Promise.all(files.map(async (f) => JSON.parse(await readFile(path.join(args.dir, f), "utf8")) as T));
  return runs.sort(byLabel);
}

function accuracy(runs: Massive[]): string {
  const pooled = table(
    ["Version", "Scenario accuracy", "Scenario ECE", "Yes/no accuracy", "Yes/no AUROC", "p50 per call", "p95 per call"],
    runs.map((r) => {
      const p = r.pooled;
      return [label(r), pct(p.scenario_accuracy), p.scenario_ece.toFixed(3), pct(p.yes_no_accuracy), p.yes_no_auroc.toFixed(3), msec(p.p50_ms), msec(p.p95_ms)];
    }),
  );
  const langs = runs[0]?.languages ?? [];
  const perLanguage = table(
    ["Language", ...runs.map((r) => `Scenario ${label(r)}`), ...runs.map((r) => `Yes/no ${label(r)}`)],
    langs.map((l) => [l, ...runs.map((r) => pct(r.per_language[l]?.scenario_accuracy)), ...runs.map((r) => pct(r.per_language[l]?.yes_no_accuracy))]),
  );
  return `${pooled}\n\n<details><summary>Per language</summary>\n\n${perLanguage}\n\n</details>`;
}

function speed(runs: Speed[]): string {
  const out: string[] = [];
  for (const checkpoint of ["english", "multilingual"]) {
    const cols = runs.filter((r) => (r.dir?.includes("multilingual") ? "multilingual" : "english") === checkpoint);
    if (!cols.length) continue;
    const cases = [...new Set(cols.flatMap((r) => r.cases.map((c) => c.case)))];
    const rows = [
      ["Load", ...cols.map((r) => msec(r.load_ms))],
      ["Memory after load (RSS)", ...cols.map((r) => `${r.rss_mb.toLocaleString("en")} MB`)],
      ...cases.map((c) => [c, ...cols.map((r) => msec(r.cases.find((x) => x.case === c)?.p50_ms))]),
    ];
    out.push(`**${checkpoint}** (p50 of each case)\n\n${table(["", ...cols.map(label)], rows)}`);
  }
  return out.join("\n\n");
}

const massive = await load<Massive>("massive-");
const speeds = await load<Speed>("speed-");
const hw = (massive[0] ?? speeds[0])?.hardware;
if (hw) console.log(`Hardware: ${hw.cpu}, ${hw.cores} cores, Node ${hw.node}\n`);
if (massive.length) console.log(`${accuracy(massive)}\n`);
if (speeds.length) console.log(speed(speeds));
