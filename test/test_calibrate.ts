import { test } from "node:test";
import assert from "node:assert/strict";
import { calibrationRecords, fitTemperature, fitTemperatureMap, type CalibrationRecord } from "../src/calibrate.js";
import { encodeRequest } from "../src/requests.js";
import type { SpecialIds } from "../src/sequence.js";

// deterministic pseudo-random numbers
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

/** Records whose labels are drawn from softmax(logits / trueT): the NLL minimizer is near trueT. */
function synthetic(trueT: number, n: number, k: number, qtype: number, seed: number): CalibrationRecord[] {
  const r = rng(seed);
  return Array.from({ length: n }, () => {
    const logits = Array.from({ length: k }, () => (r() - 0.5) * 8);
    const e = logits.map((z) => Math.exp(z / trueT));
    const sum = e.reduce((a, b) => a + b, 0);
    let u = r() * sum;
    const target = Math.max(
      0,
      e.findIndex((x) => (u -= x) <= 0),
    );
    return { qtype, k, logits, target };
  });
}

test("fitTemperature recovers the temperature the labels were drawn at", () => {
  for (const trueT of [0.7, 1.5, 3]) {
    const t = fitTemperature(synthetic(trueT, 4000, 4, 0, 7));
    assert.ok(Math.abs(t - trueT) / trueT < 0.12, `fitted ${t} for ${trueT}`);
  }
  // confined to [0.5, 5]
  assert.equal(fitTemperature(synthetic(0.1, 2000, 4, 0, 3)), 0.5);
});

test("fitTemperatureMap fits types and only well-populated buckets, and lowers the NLL", () => {
  const records = [...synthetic(2, 300, 4, 0, 1), ...synthetic(2, 40, 12, 0, 2), ...synthetic(1.2, 200, 2, 2, 3)];
  const res = fitTemperatureMap(records, { temperature: [1, 1, 1], temperature_by_options: { "choice:11+": 0.5 } }, { minPerBucket: 100 });
  assert.deepEqual(
    Object.keys(res.temperature_by_options).sort((a, b) => a.localeCompare(b)),
    ["choice:3-5", "noul:2"],
  );
  assert.deepEqual(res.n_by_bucket, { "choice:3-5": 300, "choice:11+": 40, "noul:2": 200 });
  assert.ok(res.after.nll < res.before.nll);
  assert.equal(res.temperature[1], 1); // no score records: unchanged
});

test("calibrationRecords maps labels to option indices for every question type and noul mode", () => {
  const ids: SpecialIds = { cls: 1, sep: 2, mask: 3, pad: 0, maskTok: "[MASK]" };
  const encode = (s: string) =>
    s
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => 100 + w.length);
  const config = { max_len: 64, head_max_len: 32, temperature: [1, 1, 1] as [number, number, number], temperature_by_options: {} };
  const questions = {
    c: { type: "choice", instructions: "pick", criteria: { a: null, b: null, c: null } },
    s: { type: "score", instructions: "rate", criteria: ["low", "mid", "high"] },
    n: { type: "noul", instructions: "is it" },
  } as const;
  for (const noul of ["native", "choice"] as const) {
    const { qids, items } = encodeRequest(encode, ids, config, "state", questions, { noul, optionOrders: 1 });
    const recs = calibrationRecords(
      qids,
      items,
      items.map((it) => it.markers.map(() => 0)),
      { c: "b", s: 2, n: true },
    );
    assert.deepEqual(
      recs.map((r) => r.target),
      [1, 2, noul === "native" ? 1 : 0],
    );
  }
  const { qids, items } = encodeRequest(encode, ids, config, "state", questions, { noul: "native", optionOrders: 1 });
  // unknown keys and out-of-range levels are skipped
  assert.equal(
    calibrationRecords(
      qids,
      items,
      items.map((it) => it.markers.map(() => 0)),
      { c: "zzz", s: 7 },
    ).length,
    0,
  );
});
