/**
 * End-to-end check against numbers produced by the Python reference (rl_agent_api.RLAgent.system_one)
 * on the same input. Skips when no ONNX bundle is available locally (set PED_MODEL_DIR to point at one).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import path from "node:path";
import { Ped, Router } from "../src/index.js";

const modelDir = process.env.PED_MODEL_DIR ?? path.resolve(import.meta.dirname, "../onnx");
const available = existsSync(path.join(modelDir, "ped.onnx.data"));

test("systemOne reproduces the Python reference output", { skip: !available && "no ONNX bundle on disk" }, async () => {
  const ped = await Ped.load({ modelDir });
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
  const ped = await Ped.load({ modelDir: multiDir });
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
  const router = new Router({ models: { english: { modelDir }, multilingual: { modelDir: multiDir } } });
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
