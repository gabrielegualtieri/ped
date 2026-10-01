/**
 * Temperature calibration on labeled examples, as the upstream calibrate.fit_temperature_map: one temperature
 * per question type, and one per (type, option count) bucket where there is enough data, each the
 * minimizer of the mean negative log-likelihood and confined to [TEMP_MIN, TEMP_MAX].
 *
 * The shipped checkpoints are over-confident (upstream reports ECE 0.47 -> 0.08 for English and 0.31 -> 0.11 for
 * multilingual after refitting), so probabilities should be calibrated on the task's own data before a
 * threshold is put on them.
 */
import { clampTemperature, probabilities, TEMP_MAX, TEMP_MIN, temperatureFor } from "./answers.js";
import { tempBucket } from "./sequence.js";
import type { AnswerOptions, Item } from "./requests.js";
import type { PedConfig, Question } from "./types.js";

export interface CalibrationExample<Q extends Record<string, Question> = Record<string, Question>> {
  state: unknown;
  questions: Q;
  /** the right answer per question: a choice's option key, a score's level index, a noul's true/false */
  labels: { [K in keyof Q]?: string | number | boolean };
}

export interface CalibrationOptions extends Pick<AnswerOptions, "noul"> {
  /** fewest examples for a per-option-count temperature (default 100); smaller buckets fall back to their type's */
  minPerBucket?: number;
  /** fewest examples for a per-type temperature (default 10) */
  minPerType?: number;
}

export interface CalibrationResult {
  temperature: [number, number, number];
  /** only the buckets that had enough examples; it replaces the bundle's per-option-count values */
  temperature_by_options: Record<string, number>;
  n_by_bucket: Record<string, number>;
  /** on the same examples: mean negative log-likelihood and expected calibration error (15 bins) */
  before: { nll: number; ece: number };
  after: { nll: number; ece: number };
}

export interface CalibrationRecord {
  qtype: number;
  k: number;
  logits: number[];
  target: number;
}

type Temps = Pick<PedConfig, "temperature" | "temperature_by_options">;

function targetIndex(it: Item, label: string | number | boolean): number {
  if (it.asked.t === "noul") {
    if (typeof label !== "boolean") return -1;
    // native noul options are [false, true]; asked as a choice they are [A = true, B = false]
    return it.read.t === "choice" ? Number(!label) : Number(label);
  }
  if (it.asked.t === "score") return typeof label === "number" && Number.isInteger(label) && label >= 0 && label < it.markers.length ? label : -1;
  return typeof label === "string" ? Object.keys(it.asked.crit).indexOf(label) : -1;
}

/** One record per labeled question, from sequences encoded in the order the questions define (no option orders). */
export function calibrationRecords(
  qids: string[],
  items: Item[],
  logits: ArrayLike<number>[],
  labels: Record<string, string | number | boolean | undefined>,
): CalibrationRecord[] {
  return items.flatMap((it, i) => {
    const label = labels[it.qid];
    const target = label === undefined || !qids.includes(it.qid) ? -1 : targetIndex(it, label);
    return target < 0 ? [] : [{ qtype: it.qtype, k: it.markers.length, logits: Array.from(logits[i] ?? []), target }];
  });
}

const meanNll = (recs: CalibrationRecord[], temp: (r: CalibrationRecord) => number) =>
  recs.reduce((s, r) => s - Math.log(Math.max(probabilities(r.logits, temp(r))[r.target] ?? 0, 1e-12)), 0) / Math.max(1, recs.length);

function ece(recs: CalibrationRecord[], temp: (r: CalibrationRecord) => number, bins = 15): number {
  const sums = Array.from({ length: bins }, () => ({ n: 0, conf: 0, correct: 0 }));
  for (const r of recs) {
    const p = probabilities(r.logits, temp(r));
    const conf = Math.max(...p);
    const bin = sums[Math.min(bins - 1, Math.max(0, Math.ceil(conf * bins) - 1))];
    if (!bin) continue;
    bin.n++;
    bin.conf += conf;
    bin.correct += Number(p.indexOf(conf) === r.target);
  }
  return sums.reduce((e, b) => (b.n ? e + (b.n / recs.length) * Math.abs(b.conf / b.n - b.correct / b.n) : e), 0);
}

/** The temperature minimizing the mean NLL, by golden-section search on log T (NLL is unimodal in it). */
export function fitTemperature(recs: CalibrationRecord[]): number {
  const f = (logT: number) => meanNll(recs, () => Math.exp(logT));
  const g = (Math.sqrt(5) - 1) / 2;
  let [a, b] = [Math.log(TEMP_MIN), Math.log(TEMP_MAX)];
  for (let i = 0; i < 60; i++) {
    const c = b - g * (b - a);
    const d = a + g * (b - a);
    if (f(c) < f(d)) b = d;
    else a = c;
  }
  return clampTemperature(Math.round(Math.exp((a + b) / 2) * 1e4) / 1e4);
}

export function fitTemperatureMap(records: CalibrationRecord[], current: Temps, opts: CalibrationOptions = {}): CalibrationResult {
  const minPerBucket = opts.minPerBucket ?? 100;
  const minPerType = opts.minPerType ?? 10;
  const temperature: [number, number, number] = [...current.temperature];
  for (const qt of [0, 1, 2]) {
    const recs = records.filter((r) => r.qtype === qt);
    if (recs.length >= minPerType) temperature[qt] = fitTemperature(recs);
  }
  const buckets = new Map<string, CalibrationRecord[]>();
  for (const r of records) buckets.set(tempBucket(r.qtype, r.k), [...(buckets.get(tempBucket(r.qtype, r.k)) ?? []), r]);
  const temperature_by_options = Object.fromEntries(
    Array.from(buckets).flatMap(([key, recs]) => (recs.length >= minPerBucket ? [[key, fitTemperature(recs)]] : [])),
  );
  const fitted: Temps = { temperature, temperature_by_options };
  const before = (r: CalibrationRecord) => temperatureFor(current, r.qtype, r.k);
  const after = (r: CalibrationRecord) => temperatureFor(fitted, r.qtype, r.k);
  return {
    temperature,
    temperature_by_options,
    n_by_bucket: Object.fromEntries(Array.from(buckets).map(([key, recs]) => [key, recs.length])),
    before: { nll: meanNll(records, before), ece: ece(records, before) },
    after: { nll: meanNll(records, after), ece: ece(records, after) },
  };
}
