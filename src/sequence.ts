/**
 * Pure (model-free) port of the request rendering in the checkpoint's rl_common.py:
 * option rendering, sequence layout, temperature buckets, confidence. Kept separate from the
 * ONNX session so it can be unit-tested without the weights.
 */
import type { OptionStats, Question, QuestionType, StateStats } from "./types.js";

export const QTYPES: Record<QuestionType, number> = { choice: 0, score: 1, noul: 2 };
const QTYPE_NAMES: QuestionType[] = ["choice", "score", "noul"];

/** A question as the sequence reads it (RLAgent._to_internal): a choice's list of names becomes a map. */
export type InternalQ =
  | { t: "choice"; ins: string; crit: Record<string, string | null> }
  | { t: "score"; ins: string; crit: readonly string[] }
  | { t: "noul"; ins: string; crit: { true?: string; false?: string } | undefined };

// Array.isArray narrows a readonly array to any[]; this keeps the element type
const isList = (x: unknown): x is readonly string[] => Array.isArray(x);

/** laya Agent._to_internal: an instructions object is serialized as Python's json.dumps(ensure_ascii=False) */
export function toInternal(q: Question): InternalQ {
  const ins = typeof q.instructions === "string" ? q.instructions : pyJsonDumps(q.instructions);
  if (q.type === "choice") return { t: "choice", ins, crit: isList(q.criteria) ? Object.fromEntries(q.criteria.map((c) => [c, null])) : q.criteria };
  if (q.type === "score") return { t: "score", ins, crit: q.criteria };
  return { t: "noul", ins, crit: q.criteria };
}

/** Option texts in label-index order. Noul is always [false, true] so p[1] == noul. */
export function renderOptions(q: InternalQ): string[] {
  if (q.t === "choice") return Object.entries(q.crit).map(([k, v]) => (v ? `${k}: ${v}` : k));
  if (q.t === "score") return q.crit.map((c, i) => `level ${i}: ${c}`);
  const c = q.crit ?? {};
  return ["false: " + (c.false || "no, the statement does not hold"), "true: " + (c.true || "yes, the statement holds")];
}

/** Python's json.dumps(obj, ensure_ascii=False): ", " / ": " separators, insertion key order. */
export function pyJsonDumps(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : JSON.stringify(v);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return "[" + v.map(pyJsonDumps).join(", ") + "]";
  return (
    "{" +
    Object.entries(v)
      .map(([k, x]) => `${JSON.stringify(k)}: ${pyJsonDumps(x)}`)
      .join(", ") +
    "}"
  );
}

export function serializeState(state: unknown): string {
  return typeof state === "string" ? state : pyJsonDumps(state);
}

function sizeBucket(k: number): string {
  if (k <= 2) return "2";
  if (k <= 5) return "3-5";
  if (k <= 10) return "6-10";
  return "11+";
}

/** Key for the per-cardinality temperature: a 2-option noul and a 20-option choice need different scaling. */
export function tempBucket(qtype: number, k: number): string {
  return `${QTYPE_NAMES[qtype]}:${sizeBucket(k)}`;
}

/** Jev-style confidence: 1 - normalized entropy of the answer distribution. */
export function confidenceFromProbs(p: number[]): number {
  const k = p.length;
  if (k < 2) return 1;
  let ent = 0;
  for (const x of p) ent -= x * Math.log(Math.max(x, 1e-12));
  return 1 - ent / Math.log(k);
}

export function softmax(z: number[]): number[] {
  const zmax = Math.max(...z);
  const e = z.map((v) => Math.exp(v - zmax));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export interface SpecialIds {
  cls: number;
  sep: number;
  mask: number;
  pad: number;
  /** the literal mask token text, scrubbed from user text so it cannot inject a marker */
  maskTok: string;
}

/** Tokenizer surface the sequence builder needs: text -> ids, no special tokens added. */
export type Encode = (text: string) => number[];

/** `maxLen` tokens in all, `headMaxLen` of them for the question header and its options */
export interface SequenceLimits extends SequenceOptions {
  maxLen: number;
  headMaxLen: number;
}

export interface SequenceOptions {
  /** the state's ids from `encodeState`, so a state shared by several questions is tokenized once */
  stateIds?: number[];
  /** keep the end of the state instead of its start: a conversation list ends with its newest turn */
  truncateLeft?: boolean;
  /** slot `i` shows option `order[i]` (laya `option_order`); default: the order the question defines */
  order?: number[];
}

const scrubber = (maskTok: string) => (s: string) => s.split(maskTok).join(" ");

/** The state's token ids as the sequence reads them: serialized, with the mask token scrubbed out. */
export function encodeState(encode: Encode, ids: SpecialIds, state: unknown): number[] {
  return encode(scrubber(ids.maskTok)(serializeState(state)));
}

/**
 * laya common.build_sequence:
 *   [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
 * Returns the ids, the position of each option's [MASK] marker, and what was cut to fit.
 */
export function buildSequence(
  encode: Encode,
  ids: SpecialIds,
  state: unknown,
  q: InternalQ,
  limits: SequenceLimits,
): { ids: number[]; markers: number[]; state: StateStats; options: OptionStats } {
  const { maxLen, headMaxLen, ...opts } = limits;
  const scrub = scrubber(ids.maskTok);
  let headIds = encode(`${q.t} question: ${scrub(q.ins)}`);
  const rendered = renderOptions(q);
  let optIds = (opts.order ?? rendered.map((_, i) => i)).map((i) => [ids.mask, ...encode(" " + scrub(rendered[i] ?? "")).slice(0, 48)]);
  const total = (xs: number[][]) => xs.reduce((s, o) => s + o.length, 0);
  let optBudget = headMaxLen - total(optIds);
  let perOption: number | null = null;
  if (optBudget < 16) {
    // too many / too long options: shrink every option text evenly
    perOption = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, perOption ?? o.length));
    optBudget = headMaxLen - total(optIds);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));
  const seq = [ids.cls, ...headIds, ids.sep];
  const markers: number[] = [];
  for (const o of optIds) {
    markers.push(seq.length);
    seq.push(...o);
  }
  seq.push(ids.sep);
  const room = Math.max(0, maxLen - seq.length - 1);
  const stateIds = opts.stateIds ?? encodeState(encode, ids, state);
  const st = opts.truncateLeft ? stateIds.slice(Math.max(0, stateIds.length - room)) : stateIds.slice(0, room);
  seq.push(...st, ids.sep);
  return {
    ids: seq.slice(0, maxLen),
    markers: markers.filter((m) => m < maxLen),
    state: {
      state_tokens: stateIds.length,
      state_tokens_used: st.length,
      state_tokens_dropped: stateIds.length - st.length,
      truncated: st.length < stateIds.length,
    },
    options: { options: optIds.length, options_distinct: new Set(optIds.map((o) => o.join(","))).size, tokens_per_option: perOption },
  };
}
