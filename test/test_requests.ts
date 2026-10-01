import { test } from "node:test";
import assert from "node:assert/strict";
import { aggregateWindows, clampTemperature, decodeAnswer, stateUsage } from "../src/answers.js";
import { decodeRequest, encodeRequest, optionOrders } from "../src/requests.js";
import { buildSequence, toInternal, type SpecialIds } from "../src/sequence.js";
import type { Answer, PedConfig, Usage } from "../src/types.js";

const ids: SpecialIds = { cls: 1, sep: 2, mask: 3, pad: 0, maskTok: "[MASK]" };
// one id per whitespace-separated word; the id encodes the word length so tests can recognise words
const encode = (s: string) =>
  s
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => 100 + w.length);
const config: PedConfig = { max_len: 64, head_max_len: 32, temperature: [1, 1, 1], temperature_by_options: {} };

test("buildSequence reuses given state ids, cuts a conversation from the left and reports the cut", () => {
  const q = toInternal({ type: "noul", instructions: "is it" });
  const stateIds = Array.from({ length: 100 }, (_, i) => 1000 + i);
  const right = buildSequence(encode, ids, "ignored", q, { maxLen: 64, headMaxLen: 32, stateIds });
  const left = buildSequence(encode, ids, "ignored", q, { maxLen: 64, headMaxLen: 32, stateIds, truncateLeft: true });
  const room = right.state.state_tokens_used;
  assert.deepEqual(right.ids.slice(-room - 1, -1), stateIds.slice(0, room));
  assert.deepEqual(left.ids.slice(-room - 1, -1), stateIds.slice(-room));
  assert.deepEqual(right.state, { state_tokens: 100, state_tokens_used: room, state_tokens_dropped: 100 - room, truncated: true });
});

test("buildSequence places options in the given order and reports collapsed options", () => {
  const q = toInternal({ type: "choice", instructions: "pick", criteria: ["a", "bbb"] });
  const s = buildSequence(encode, ids, "x", q, { maxLen: 64, headMaxLen: 32, order: [1, 0] });
  assert.equal(s.ids[(s.markers[0] ?? 0) + 1], 103); // "bbb" first
  assert.equal(s.ids[(s.markers[1] ?? 0) + 1], 101);
  assert.deepEqual(s.options, { options: 2, options_distinct: 2, tokens_per_option: null });
  // 20 options that share their first words collapse once the head budget cuts them to 4 tokens
  const many = toInternal({ type: "choice", instructions: "pick", criteria: Array.from({ length: 20 }, (_, i) => `same same same same ${"x".repeat(i + 1)}`) });
  const m = buildSequence(encode, ids, "x", many, { maxLen: 512, headMaxLen: 64 });
  assert.equal(m.options.tokens_per_option, 4);
  assert.equal(m.options.options_distinct, 1);
});

test("optionOrders: as defined first, reversed second, all distinct, never more than exist", () => {
  const o = optionOrders(5, 4);
  assert.deepEqual(o[0], [0, 1, 2, 3, 4]);
  assert.deepEqual(o[1], [4, 3, 2, 1, 0]);
  assert.equal(new Set(o.map((x) => x.join())).size, 4);
  for (const x of o)
    assert.deepEqual(
      [...x].sort((a, b) => a - b),
      [0, 1, 2, 3, 4],
    );
  assert.deepEqual(optionOrders(2, 5), [
    [0, 1],
    [1, 0],
  ]);
  assert.deepEqual(optionOrders(1, 3), [[0]]);
  assert.deepEqual(optionOrders(4, 3), optionOrders(4, 3));
});

test("encodeRequest: one sequence per order, scores keep their order, a noul can be asked as a choice", () => {
  const questions = {
    c: { type: "choice", instructions: "pick", criteria: ["a", "b", "c"] },
    s: { type: "score", instructions: "rate", criteria: ["low", "high"] },
    n: { type: "noul", instructions: "is it" },
  } as const;
  const { items } = encodeRequest(encode, ids, config, "some state", questions, { noul: "choice", optionOrders: 3 });
  assert.deepEqual(
    items.map((it) => it.qid),
    ["c", "c", "c", "s", "n", "n"],
  );
  const n = items.find((it) => it.qid === "n");
  assert.equal(n?.read.t, "choice");
  assert.deepEqual(n?.read.crit, { A: "yes", B: "no" });
  assert.throws(() => encodeRequest(encode, ids, config, "x", {}, { noul: "native", optionOrders: 1 }), /at least one question/);
});

test("decodeRequest averages the orders back in the defined order and maps a choice-asked noul to P(true)", () => {
  const questions = {
    c: { type: "choice", instructions: "pick", criteria: ["a", "b"] },
    n: { type: "noul", instructions: "is it" },
  } as const;
  const { qids, items } = encodeRequest(encode, ids, config, "state", questions, { noul: "choice", optionOrders: 2 });
  // c: order [0,1] says a, order [1,0] says a too (slot 1 is a); n: A (= true) strongly
  const rows = [
    { logits: [2, 0], act: 1 },
    { logits: [0, 2], act: 1 },
    { logits: [3, 0], act: 1 },
    { logits: [0, 3], act: 1 },
  ];
  const { answers, usage } = decodeRequest(config, qids, items, rows);
  const c = answers["c"];
  assert.equal(c?.type === "choice" && c.choice, "a");
  assert.equal(c?.type === "choice" && c.probabilities["a"], 0.8808);
  const n = answers["n"];
  assert.equal(n?.type === "noul" && n.noul, 0.9526);
  assert.equal(
    usage.input_tokens,
    items.reduce((s, it) => s + it.ids.length, 0),
  );
});

test("decodeAnswer reports answer_confidence on every type; temperatures are clamped to [0.5, 5]", () => {
  const n = decodeAnswer(toInternal({ type: "noul", instructions: "x" }), [0.3, 0.7], 1);
  assert.deepEqual(n, { type: "noul", noul: 0.7, confidence: 0.7, answer_confidence: 0.7, rl_agent: { act_probability: 1 } });
  const c = decodeAnswer(toInternal({ type: "choice", instructions: "x", criteria: ["a", "b", "c"] }), [0.2, 0.5, 0.3], 1);
  assert.equal(c.type === "choice" && c.answer_confidence, 0.5);
  assert.equal(clampTemperature(0.1006), 0.5);
  assert.equal(clampTemperature(12), 5);
  assert.equal(clampTemperature(1.76), 1.76);
  assert.equal(clampTemperature(Number.NaN), 1);
});

test("aggregateWindows: noul takes the highest P(true), choice the most confident window, ties the first", () => {
  const usage = (truncated: boolean): Usage => ({
    input_tokens: 10,
    output_tokens: 0,
    state_tokens: 5,
    state_tokens_dropped: truncated ? 2 : 0,
    truncated,
    truncated_questions: truncated ? ["c"] : [],
  });
  const noul = (p: number): Answer => ({
    type: "noul",
    noul: p,
    confidence: Math.max(p, 1 - p),
    answer_confidence: Math.max(p, 1 - p),
    rl_agent: { act_probability: 1 },
  });
  const choice = (best: string, conf: number): Answer => ({
    type: "choice",
    choice: best,
    probabilities: {},
    confidence: 0,
    answer_confidence: conf,
    rl_agent: { act_probability: 1 },
  });
  const results = [
    { answers: { n: noul(0.1), c: choice("x", 0.6) }, usage: usage(false) },
    { answers: { n: noul(0.8), c: choice("y", 0.9) }, usage: usage(true) },
    { answers: { n: noul(0.3), c: choice("z", 0.9) }, usage: usage(false) },
  ];
  const spans: [number, number][] = [
    [0, 100],
    [50, 150],
    [100, 180],
  ];
  const { answers, usage: u } = aggregateWindows(["n", "c"], results, spans, 180);
  assert.equal(answers["n"]?.type === "noul" && answers["n"].noul, 0.8);
  assert.deepEqual(answers["n"]?.window, { index: 1, token_start: 50, token_end: 150, count: 3 });
  assert.equal(answers["c"]?.type === "choice" && answers["c"].choice, "y");
  assert.deepEqual(u, {
    input_tokens: 30,
    output_tokens: 0,
    state_tokens: 180,
    state_tokens_dropped: 2,
    truncated: 1,
    truncated_questions: ["c"],
    windows: 3,
  });
});

test("stateUsage reports the worst truncation and only collapsed questions", () => {
  const stat = (dropped: number, distinct: number) => ({
    ids: [1, 2, 3],
    state: { state_tokens: 10, state_tokens_used: 10 - dropped, state_tokens_dropped: dropped, truncated: dropped > 0 },
    options: { options: 3, options_distinct: distinct, tokens_per_option: distinct < 3 ? 4 : null },
  });
  const u = stateUsage(["a", "b"], [stat(0, 3), stat(4, 2)]);
  assert.equal(u.truncated, true);
  assert.equal(u.state_tokens_dropped, 4);
  assert.deepEqual(u.truncated_questions, ["b"]);
  assert.deepEqual(u.options, { b: { options: 3, options_distinct: 2, tokens_per_option: 4 } });
});
