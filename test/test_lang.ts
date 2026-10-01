import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { analyse, detectScript, isEnglish } from "../src/lang.js";
import { cpLength, cpSlice, isPyUpper, pySplit, pyStrip } from "../src/lang-latin.js";

interface Golden {
  state: unknown;
  script: string;
  language: string | null;
  isEnglish: boolean;
  languageUndecided: boolean;
  mixedSegment: string | null;
}

// Decisions of the upstream lang.py `analyse` (upstream 0.3.22) on hand-written support messages in ~30 languages
// and on generated states mixing function words, scripts, identifiers, code, acronyms and fields.
// A differential run of the full 5,570-state corpus agreed on every field.
const isGoldenList = (x: unknown): x is Golden[] => Array.isArray(x) && x.every((g) => typeof g === "object" && g !== null && "state" in g && "script" in g);
const parsed: unknown = JSON.parse(readFileSync(new URL("fixtures/lang-golden.json", import.meta.url), "utf8"));
if (!isGoldenList(parsed)) throw new Error("fixtures/lang-golden.json: unexpected shape");
const golden = parsed;

test("analyse matches the Python reference on the golden corpus", () => {
  const wrong = golden.flatMap((g) => {
    const d = analyse(g.state);
    const got = { script: d.script, language: d.language, isEnglish: d.isEnglish, languageUndecided: d.languageUndecided, mixedSegment: d.mixedSegment };
    const want = { script: g.script, language: g.language, isEnglish: g.isEnglish, languageUndecided: g.languageUndecided, mixedSegment: g.mixedSegment };
    return JSON.stringify(got) === JSON.stringify(want) ? [] : [{ state: g.state, got, want }];
  });
  assert.equal(golden.length, 772);
  assert.deepEqual(wrong, []);
});

test("analyse: the cases the heuristics were built for", () => {
  // script decides first: the English checkpoint cannot read these at all
  assert.equal(analyse("我的包裹已经晚了一个星期，请帮我查一下。").script, "han");
  assert.equal(isEnglish("Мне дважды списали деньги, верните их немедленно!"), false);
  // Latin-script languages by function words, accents or not
  assert.equal(analyse("Il pacco è arrivato rotto, voglio i soldi indietro!").language, "it");
  assert.equal(analyse("Voce pode me mandar a nota fiscal?").language, "pt");
  assert.equal(analyse("Deu erro 500 no endpoint de login depois do update").language, "pt");
  // English stays English despite a loanword, a symbol, a capitalised name, links and repeated collisions
  assert.equal(isEnglish("My café order: the latte was cold and the croissant was stale, please refund it"), true);
  assert.equal(isEnglish("Set α to 0.05 in the config and rerun the test"), true);
  assert.equal(isEnglish("The customer Дмитрий called about the order and wants a refund today"), true);
  assert.equal(isEnglish("see github.com and user@acme.com for v1.2.3 of the U.S.A. release"), true);
  assert.equal(isEnglish("do more, do less, do it now"), true);
  // too short to tell: undecided, which the router sends to its default
  const short = analyse("Grazie mille");
  assert.equal(short.language, null);
  assert.equal(short.languageUndecided, true);
  // no letters at all
  assert.equal(analyse("12345").script, "unknown");
  assert.equal(analyse(null).script, "unknown");
});

test("analyse: one non-English line or field is enough", () => {
  const ticket = {
    subject: "Payment failed on checkout page for the customer account",
    log: "Traceback (most recent call last):\n  File pay.py, line 10\nKeyError: card",
    message: "Olá, o sistema deu erro quando tentei pagar com o meu cartão",
  };
  const d = analyse(ticket);
  assert.equal(d.isEnglish, false);
  assert.equal(d.language, "pt");
  const mixed = analyse(
    "We have received the following note from the customer about the order that was delivered to the wrong address last week, " +
      "and we need to decide what to do with it before the end of the day.\nO pacote não chegou e eu quero o meu dinheiro de volta",
  );
  assert.equal(mixed.isEnglish, false);
  assert.equal(mixed.mixedSegment, "O pacote não chegou e eu quero o meu dinheiro de volta");
});

test("detectScript and the Python string helpers", () => {
  assert.equal(detectScript("注文した商品"), "han");
  assert.equal(detectScript("ありがとう"), "kana");
  assert.equal(detectScript("!!!"), "unknown");
  assert.equal(cpLength("a😡b"), 3);
  assert.equal(cpSlice("😡😡😡", 2), "😡😡");
  assert.deepEqual(pySplit("\x1cfield\x1dsep  x y"), ["field", "sep", "x", "y"]);
  assert.equal(pyStrip("﻿ x \x85"), "﻿ x");
  assert.equal(isPyUpper("MON"), true);
  assert.equal(isPyUpper("Mon"), false);
  assert.equal(isPyUpper("ABC中"), true);
  assert.equal(isPyUpper("123"), false);
});
