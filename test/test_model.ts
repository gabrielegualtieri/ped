/**
 * End-to-end check against numbers produced by the Python reference (rl_agent_api.RLAgent.system_one)
 * on the same input. Skips when no ONNX bundle is available locally (set PED_MODEL_DIR to point at one).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import path from "node:path";
import { Ped, Router, type AnswerOptions } from "../src/index.js";

// JavaScript callers can pass anything: build such input without a type assertion
const untyped = <T>(x: unknown): x is T => x !== undefined;

const modelDir = process.env.PED_MODEL_DIR ?? path.resolve(import.meta.dirname, "../onnx");
const available = existsSync(path.join(modelDir, "ped.onnx.data"));

test("systemOne reproduces the Python reference output", { skip: !available && "no ONNX bundle on disk" }, async () => {
  // the reference's own noul head: the default asks yes/no as a two-option choice
  const ped = await Ped.load({ modelDir, noul: "native" });
  try {
    const r = await ped.systemOne(
      {
        subject: "Refund not received",
        body: "I cancelled my subscription two weeks ago and I still have not received my refund. This is the third time I am writing. If this is not resolved I will dispute the charge with my bank.",
      },
      {
        department: {
          type: "choice",
          instructions: "Which team should handle this ticket?",
          criteria: { billing: "payments, refunds, invoices", support: "product help and bugs", sales: "new purchases and upgrades" },
        },
        urgency: { type: "score", instructions: "How urgent is this ticket?", criteria: ["not urgent", "somewhat urgent", "urgent", "critical"] },
        churn_risk: { type: "noul", instructions: "Is the customer likely to cancel or dispute?" },
      },
    );
    assert.equal(r.usage.input_tokens, 267);
    assert.equal(r.answers.department.choice, "billing");
    assert.deepEqual(r.answers.department.probabilities, { billing: 0.9415, support: 0.031, sales: 0.0275 });
    assert.equal(r.answers.department.confidence, 0.7603);
    assert.equal(r.answers.urgency.score, 1.3886);
    assert.deepEqual(r.answers.urgency.probabilities, { "0": 0.1752, "1": 0.2947, "2": 0.4962, "3": 0.0338 });
    assert.equal(r.answers.churn_risk.noul, 0.0988);
  } finally {
    await ped.close();
  }
});

// Multilingual checkpoint (mmBERT): reference numbers from rl_agent_api.RLAgent.system_one on
// convaiinnovations/laya/multilingual. The e-mail state pins the tokenizer's handling of runs of spaces.
const multiDir = process.env.PED_MODEL_DIR_MULTILINGUAL ?? path.resolve(import.meta.dirname, "../onnx/multilingual");
const multiAvailable = existsSync(path.join(multiDir, "ped.onnx.data"));

test("multilingual systemOne reproduces the Python reference output", { skip: !multiAvailable && "no multilingual ONNX bundle on disk" }, async () => {
  const ped = await Ped.load({ modelDir: multiDir, noul: "native" });
  try {
    const questions = {
      reparto: {
        type: "choice",
        instructions: "Quale reparto deve gestire questo messaggio?",
        criteria: {
          rimborsi: "soldi indietro, resi, pagamenti",
          spedizioni: "consegne, corrieri, pacchi in ritardo",
          vendite: "preventivi, prezzi, nuovi acquisti",
        },
      },
      arrabbiato: { type: "noul", instructions: "Il cliente è arrabbiato?", criteria: { true: "sì", false: "no" } },
      lista: { type: "choice", instructions: "Lingua del testo", criteria: ["italiano", "inglese", "tedesco"] },
    } as const;
    const r = await ped.systemOne("Il pacco è arrivato rotto, voglio i soldi indietro!", questions);
    assert.equal(r.usage.input_tokens, 135);
    assert.deepEqual(r.answers.reparto.probabilities, { rimborsi: 0.7792, spedizioni: 0.2164, vendite: 0.0044 });
    assert.equal(r.answers.arrabbiato.noul, 0.3474);
    assert.deepEqual(r.answers.lista.probabilities, { italiano: 0.5103, inglese: 0.253, tedesco: 0.2367 });

    const mail = "Buongiorno,\n\n   ho un problema  con l'ordine n. 12345.   Il pacco  non è arrivato.\n\nGrazie,\n  Luca";
    const m = await ped.systemOne(mail, questions);
    assert.equal(m.usage.input_tokens, 213);
    assert.deepEqual(m.answers.reparto.probabilities, { rimborsi: 0.0004, spedizioni: 0.9989, vendite: 0.0007 });
    assert.equal(m.answers.arrabbiato.noul, 0.0601);
    assert.deepEqual(m.answers.lista.probabilities, { italiano: 0.6798, inglese: 0.1651, tedesco: 0.1552 });
  } finally {
    await ped.close();
  }
});

test("Router sends English and Italian to their checkpoints", { skip: !(available && multiAvailable) && "both ONNX bundles needed" }, async () => {
  const router = new Router({ models: { english: { modelDir }, multilingual: { modelDir: multiDir } }, noul: "native" });
  try {
    const q = { angry: { type: "noul", instructions: "Is the customer angry?" } } as const;
    const en = await router.systemOne("I cancelled two weeks ago and I still have no refund, this is unacceptable.", q);
    const it = await router.systemOne("Il pacco è arrivato rotto, voglio i soldi indietro!", q);
    assert.equal(en.routing.model, "english");
    assert.equal(it.routing.model, "multilingual");
    assert.equal(it.answers.angry.noul, 0.7234);
  } finally {
    await router.close();
  }
});

test("systemOneBatch answers each state as systemOne does", { skip: !multiAvailable && "no multilingual ONNX bundle on disk" }, async () => {
  const ped = await Ped.load({ modelDir: multiDir });
  try {
    const questions = {
      team: {
        type: "choice",
        instructions: "Which team should handle this message?",
        criteria: { billing: "refunds, charges", shipping: "deliveries, tracking" },
      },
      angry: { type: "noul", instructions: "Is the customer angry?" },
    } as const;
    const states = [
      "Il pacco è arrivato rotto, voglio i soldi indietro!",
      "Where is my parcel? Tracking has not moved for a week.",
      { body: "Mi avete addebitato due volte" },
    ];
    const batch = await ped.systemOneBatch(states, questions, { batchSize: 3 });
    for (const [i, state] of states.entries()) {
      const one = await ped.systemOne(state, questions);
      const b = batch[i];
      assert.ok(b);
      assert.equal(b.answers.team.choice, one.answers.team.choice);
      for (const k of ["billing", "shipping"] as const)
        assert.ok(Math.abs((b.answers.team.probabilities[k] ?? 0) - (one.answers.team.probabilities[k] ?? 0)) <= 2e-4);
      assert.ok(Math.abs(b.answers.angry.noul - one.answers.angry.noul) <= 2e-4);
      assert.deepEqual(b.usage, one.usage);
    }
  } finally {
    await ped.close();
  }
});

test("systemOneLong reads a document past the sequence limit", { skip: !multiAvailable && "no multilingual ONNX bundle on disk" }, async () => {
  const ped = await Ped.load({ modelDir: multiDir, maxLen: 320 });
  try {
    const filler = "The quarterly newsletter covers the new office opening, the summer party and the parking rules. ";
    const doc = filler.repeat(20) + "Separately: I was charged twice for my subscription this month and I want a refund of the second charge.";
    const q = { refund: { type: "noul", instructions: "Does the customer ask for a refund?" } } as const;
    const cut = await ped.systemOne(doc, q);
    assert.equal(cut.usage.truncated, true);
    const long = await ped.systemOneLong(doc, q);
    assert.ok(long.usage.windows > 2, `windows ${long.usage.windows}`);
    const w = long.answers.refund.window;
    assert.ok(w && w.token_end === long.usage.state_tokens, "the deciding window is the one holding the request");
    assert.ok(long.answers.refund.noul > cut.answers.refund.noul + 0.2, `long ${long.answers.refund.noul} vs truncated ${cut.answers.refund.noul}`);
    // without the request no window may claim it (asked as a two-option choice, one of these windows gets 0.99)
    const none = await ped.systemOneLong(filler.repeat(20) + "Separately: thanks for the update, see you at the party.", q);
    assert.ok(none.answers.refund.noul < 0.1, `no request: ${none.answers.refund.noul}`);
    // a state that fits one window: systemOne's answers with the same noul mode, one window, no window attribution
    const short = await ped.systemOneLong("I want a refund", q);
    const one = await ped.systemOne("I want a refund", q, { noul: "native" });
    assert.equal(short.usage.windows, 1);
    assert.equal(short.answers.refund.noul, one.answers.refund.noul);
    assert.equal(short.answers.refund.window, undefined);
  } finally {
    await ped.close();
  }
});

test("option orders and noul-as-choice keep the answer shapes", { skip: !multiAvailable && "no multilingual ONNX bundle on disk" }, async () => {
  const ped = await Ped.load({ modelDir: multiDir, noul: "choice", optionOrders: 3 });
  try {
    const r = await ped.systemOne("Il pacco è arrivato rotto, voglio i soldi indietro!", {
      team: { type: "choice", instructions: "Which team?", criteria: { billing: "refunds", shipping: "deliveries", sales: "prices" } },
      level: { type: "score", instructions: "How urgent?", criteria: ["low", "medium", "high"] },
      angry: { type: "noul", instructions: "Is the customer angry?" },
    });
    const sum = Object.values(r.answers.team.probabilities).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-3);
    assert.equal(r.answers.angry.type, "noul");
    assert.ok(r.answers.angry.noul >= 0 && r.answers.angry.noul <= 1);
    assert.equal(r.answers.level.type, "score");
    // 3 orders for the choice, 1 for the ordinal score, 2 (all there are) for the yes/no
    const native = await ped.systemOne("x", { a: { type: "noul", instructions: "Is it?" } }, { noul: "native", optionOrders: 1 });
    assert.equal(native.answers.a.type, "noul");
    const maybe: unknown = { noul: "maybe" };
    if (untyped<AnswerOptions>(maybe))
      await assert.rejects(ped.systemOne("x", { a: { type: "noul", instructions: "Is it?" } }, maybe), /noul must be one of choice, native/);
  } finally {
    await ped.close();
  }
});
