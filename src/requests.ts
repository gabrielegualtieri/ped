/**
 * One request (a state and its questions) -> the sequences the model reads, and their rows -> answers.
 *
 * The state is tokenized once and shared by every question. A conversation list keeps its newest turns
 * when it has to be cut. Each question becomes one sequence per option order (`optionOrders`): the
 * answer averages the probabilities of the orders, which removes the position bias of a single order.
 * With `noul: "choice"` a yes/no question is put to the model as a two-option choice with neutral keys,
 * the workaround upstream documents for the noul head answering its false/true labels instead of the state
 * (upstream issue #156); the answer keeps the noul shape.
 */
import { decodeAnswer, probabilities, stateUsage, temperatureFor } from "./answers.js";
import { buildSequence, encodeState, QTYPES, renderOptions, toInternal, type Encode, type InternalQ, type SpecialIds } from "./sequence.js";
import type { Answer, OptionStats, PedConfig, Question, StateStats, Usage } from "./types.js";

export type NoulMode = "native" | "choice";

export interface AnswerOptions {
  /**
   * how a yes/no question is asked: "choice", a two-option choice (the default of `systemOne` and
   * `systemOneBatch`); "native", the checkpoint's noul head (the default of `systemOneLong`)
   */
  noul?: NoulMode;
  /** option orders averaged per choice / noul question; 1 asks each question once */
  optionOrders?: number;
}

export interface Item {
  qid: string;
  /** the question as the caller asked it */
  asked: InternalQ;
  /** the question as the model reads it (a noul asked as a choice differs) */
  read: InternalQ;
  qtype: number;
  ids: number[];
  markers: number[];
  /** slot i shows option order[i] */
  order: number[];
  state: StateStats;
  options: OptionStats;
}

export interface Row {
  logits: ArrayLike<number>;
  act: number;
}

function asChoice(q: Extract<InternalQ, { t: "noul" }>): InternalQ {
  return { t: "choice", ins: q.ins, crit: { A: q.crit?.true || "yes", B: q.crit?.false || "no" } };
}

function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** `n` option orders of `k` options, all distinct where possible: as defined, reversed, then seeded shuffles. */
export function optionOrders(k: number, n: number): number[][] {
  const identity = Array.from({ length: k }, (_, i) => i);
  const orders = [identity, [...identity].reverse()];
  const rnd = seeded(k * 7919 + 17);
  for (let tries = 0; orders.length < n && tries < 50 * n; tries++) {
    const o = [...identity];
    for (let i = k - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const [x, y] = [o[i] ?? i, o[j] ?? j];
      o[i] = y;
      o[j] = x;
    }
    if (!orders.some((x) => x.join() === o.join())) orders.push(o);
  }
  // k = 1 or 2 has fewer distinct orders than asked for
  return orders.filter((o, i) => orders.findIndex((x) => x.join() === o.join()) === i).slice(0, Math.max(1, n));
}

/** The sequences for one state: tokenized once, one sequence per question and option order. */
export function encodeRequest(
  encode: Encode,
  ids: SpecialIds,
  config: PedConfig,
  state: unknown,
  questions: Record<string, Question>,
  opts: Required<AnswerOptions>,
): { qids: string[]; items: Item[] } {
  const qids = Object.keys(questions);
  if (qids.length === 0) throw new Error("systemOne: at least one question is required");
  const stateIds = encodeState(encode, ids, state);
  // a chronological conversation list is serialized newest-last: cut it from the left
  const truncateLeft = Array.isArray(state);
  const items: Item[] = [];
  for (const [qid, question] of Object.entries(questions)) {
    const asked = toInternal(question);
    const read = asked.t === "noul" && opts.noul === "choice" ? asChoice(asked) : asked;
    const k = renderOptions(read).length;
    // a score's levels are ordinal: their order is part of the question
    const orders = read.t === "score" ? optionOrders(k, 1) : optionOrders(k, opts.optionOrders);
    for (const order of orders) {
      const s = buildSequence(encode, ids, state, read, { maxLen: config.max_len, headMaxLen: config.head_max_len, stateIds, truncateLeft, order });
      if (s.markers.length !== k) {
        throw new Error(
          `question ${JSON.stringify(qid)}: only ${s.markers.length} of its ${k} option markers fit in max_len=${config.max_len} ` +
            `with head_max_len=${config.head_max_len} spent on the question; lower head_max_len, raise max_len, or use fewer options`,
        );
      }
      items.push({ qid, asked, read, qtype: QTYPES[read.t], ids: s.ids, markers: s.markers, order, state: s.state, options: s.options });
    }
  }
  return { qids, items };
}

/** Probabilities in the order the question defines its options, averaged over the option orders. */
function averagedProbabilities(config: PedConfig, mine: [Item, Row][]): number[] {
  const k = mine[0]?.[0].markers.length ?? 0;
  const avg = new Array<number>(k).fill(0);
  for (const [it, row] of mine) {
    const p = probabilities(row.logits, temperatureFor(config, it.qtype, k));
    it.order.forEach((opt, slot) => (avg[opt] = (avg[opt] ?? 0) + (p[slot] ?? 0) / mine.length));
  }
  return avg;
}

/** One state's rows (one per item, in item order) -> its answers and usage. */
export function decodeRequest(config: PedConfig, qids: string[], items: Item[], rows: Row[]): { answers: Record<string, Answer>; usage: Usage } {
  const answers: Record<string, Answer> = {};
  const firsts: Item[] = [];
  for (const qid of qids) {
    const mine = items.flatMap((it, i): [Item, Row][] => (it.qid === qid && rows[i] ? [[it, rows[i]]] : []));
    const [first, firstRow] = mine[0] ?? [];
    if (!first || !firstRow) continue;
    firsts.push(first);
    const p = averagedProbabilities(config, mine);
    // a noul asked as a choice: option A is true, B is false -> noul's [false, true]
    const asked = first.asked.t === "noul" && first.read.t === "choice" ? [p[1] ?? 0, p[0] ?? 0] : p;
    answers[qid] = decodeAnswer(first.asked, asked, firstRow.act);
  }
  const usage = stateUsage(qids, firsts);
  usage.input_tokens = items.reduce((s, it) => s + it.ids.length, 0);
  return { answers, usage };
}
