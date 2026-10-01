/**
 * Model rows -> typed answers, and the per-question aggregation of a long document's windows. Kept apart
 * from the ONNX session so both can be unit-tested without the weights.
 */
import { confidenceFromProbs, softmax, tempBucket, type InternalQ } from "./sequence.js";
import type { Answer, LongUsage, OptionStats, PedConfig, StateStats, Usage } from "./types.js";

const round4 = (x: number) => Math.round(x * 1e4) / 1e4;

// laya clamps every temperature into this range at load: a value outside it (the English checkpoint's
// choice:11+ is 0.10, a 10x sharpener that returns point masses, laya issue #394) is not usable as is
export const TEMP_MIN = 0.5;
export const TEMP_MAX = 5.0;

/** A usable temperature: `t` confined to [TEMP_MIN, TEMP_MAX], 1.0 when it is not a finite number. */
export function clampTemperature(t: unknown): number {
  return typeof t === "number" && Number.isFinite(t) ? Math.min(TEMP_MAX, Math.max(TEMP_MIN, t)) : 1;
}

/** The temperatures `config` asks for, each clamped; the per-cardinality bucket wins over the per-type one. */
export function clampedTemperatures(
  config: Pick<PedConfig, "temperature" | "temperature_by_options">,
): Pick<PedConfig, "temperature" | "temperature_by_options"> {
  return {
    temperature: [clampTemperature(config.temperature[0]), clampTemperature(config.temperature[1]), clampTemperature(config.temperature[2])],
    temperature_by_options: Object.fromEntries(Object.entries(config.temperature_by_options).map(([k, v]) => [k, clampTemperature(v)])),
  };
}

export function temperatureFor(config: Pick<PedConfig, "temperature" | "temperature_by_options">, qtype: number, k: number): number {
  return config.temperature_by_options[tempBucket(qtype, k)] ?? config.temperature[qtype] ?? 1;
}

/** One question's answer from its option probabilities (rl_agent_api / laya `_decode_answers`). */
export function decodeAnswer(q: InternalQ, p: number[], actProbability: number): Answer {
  const rl_agent = { act_probability: actProbability };
  // max(p): the quantity temperature scaling fits, reported on every type so callers can gate on one number
  const answer_confidence = round4(Math.max(...p));
  if (q.t === "choice") {
    const keys = Object.keys(q.crit);
    const best = p.indexOf(Math.max(...p));
    return {
      type: "choice",
      choice: keys[best] ?? "",
      probabilities: Object.fromEntries(keys.map((kk, i) => [kk, round4(p[i] ?? 0)])),
      confidence: round4(confidenceFromProbs(p)),
      answer_confidence,
      rl_agent,
    };
  }
  if (q.t === "score") {
    return {
      type: "score",
      score: round4(p.reduce((s, v, i) => s + i * v, 0)),
      legend: Object.fromEntries(q.crit.map((c, i) => [String(i), c])),
      probabilities: Object.fromEntries(p.map((v, i) => [String(i), round4(v)])),
      confidence: round4(confidenceFromProbs(p)),
      answer_confidence,
      rl_agent,
    };
  }
  const t = p[1] ?? 0;
  return { type: "noul", noul: round4(t), confidence: round4(Math.max(t, 1 - t)), answer_confidence, rl_agent };
}

/** Option probabilities from raw logits at temperature `temp`. */
export const probabilities = (logits: ArrayLike<number>, temp: number): number[] => softmax(Array.from(logits, (v) => v / temp));

/** One state's usage: tokens, and the truncation and option-collapse reports laya returns. */
export function stateUsage(qids: string[], items: { ids: number[]; state: StateStats; options: OptionStats }[]): Usage {
  const dropped = Math.max(...items.map((it) => it.state.state_tokens_dropped));
  const usage: Usage = {
    input_tokens: items.reduce((s, it) => s + it.ids.length, 0),
    output_tokens: 0,
    state_tokens: items[0]?.state.state_tokens ?? 0,
    // worst case: the questions share one state, not one head budget
    state_tokens_dropped: dropped,
    truncated: dropped > 0,
    truncated_questions: qids.filter((_, i) => items[i]?.state.truncated),
  };
  const collapsed = Object.fromEntries(
    qids.flatMap((qid, i) => (items[i] && items[i].options.options_distinct < items[i].options.options ? [[qid, items[i].options]] : [])),
  );
  if (Object.keys(collapsed).length > 0) usage.options = collapsed;
  return usage;
}

/**
 * laya `predict_long`'s "auto" aggregation over a long document's windows, per question:
 *   noul          -> the window with the highest P(true): the statement holds if any window supports it
 *   choice, score -> the most confident window, so a localized signal is not out-voted by neutral text
 * The returned probabilities are the deciding window's, not a calibrated number for the whole document.
 */
export function aggregateWindows(
  qids: string[],
  results: { answers: Record<string, Answer>; usage: Usage }[],
  spans: [number, number][],
  documentTokens: number,
): { answers: Record<string, Answer>; usage: LongUsage } {
  const answers: Record<string, Answer> = {};
  const score = (a: Answer) => (a.type === "noul" ? a.noul : a.answer_confidence);
  for (const qid of qids) {
    const per = results.map((r) => r.answers[qid]).filter((a): a is Answer => a !== undefined);
    // first window wins a tie, as Python's max() does
    const best = per.reduce((b, a, j) => (score(a) > score(per[b] ?? a) ? j : b), 0);
    const [start, end] = spans[best] ?? [0, 0];
    const answer = per[best];
    if (answer) answers[qid] = { ...answer, window: { index: best, token_start: start, token_end: end, count: results.length } };
  }
  const usages = results.map((r) => r.usage);
  const sum = (f: (u: Usage) => number) => usages.reduce((s, u) => s + f(u), 0);
  const options = usages.reduce<Record<string, OptionStats> | undefined>((acc, u) => (u.options ? { ...acc, ...u.options } : acc), undefined);
  return {
    answers,
    usage: {
      input_tokens: sum((u) => u.input_tokens),
      output_tokens: 0,
      state_tokens: documentTokens,
      state_tokens_dropped: sum((u) => u.state_tokens_dropped),
      truncated: sum((u) => Number(u.truncated)),
      truncated_questions: [...new Set(usages.flatMap((u) => u.truncated_questions))],
      ...(options ? { options } : {}),
      windows: results.length,
    },
  };
}

export { round4 };
