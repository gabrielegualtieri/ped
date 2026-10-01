import { test } from "node:test";
import assert from "node:assert/strict";
import { englishFromCode, route, Router, type RouteOptions, type RouterOptions } from "../src/router.js";

// JavaScript callers can pass anything: build such input without a type assertion
const untyped = <T>(x: unknown): x is T => x !== undefined;

test("route: English text to the English checkpoint, everything else to the multilingual one", () => {
  assert.equal(route("I was charged twice this month, please refund me today").model, "english");
  assert.equal(route("Il pacco è arrivato rotto, voglio i soldi indietro!").model, "multilingual");
  assert.equal(route("Quero cancelar meu plano agora").model, "multilingual");
  assert.equal(route({ subject: "Refund", body: "Ho annullato e non ho ricevuto il rimborso" }).model, "multilingual");
  const zh = route("我的包裹已经晚了一个星期");
  assert.equal(zh.model, "multilingual");
  assert.match(zh.reason, /^non-Latin script \(han, 100% of letters\)/);
  assert.equal(zh.detection?.script, "han");
});

test("route: a state with no clue goes to the default, multilingual unless set", () => {
  for (const s of ["Grazie mille", "Thanks", "", "12345", null]) assert.equal(route(s).model, "multilingual", JSON.stringify(s));
  assert.equal(route("Thanks", { default: "english" }).model, "english");
  assert.match(route("Thanks").reason, /using default \(multilingual\)$/);
});

test("route: explicit model and language win over detection", () => {
  const it = "Il pacco è arrivato rotto, voglio i soldi indietro!";
  assert.deepEqual(route(it, { model: "english" }), { model: "english", reason: "explicit model=english", detection: null });
  assert.equal(route(it, { lang: "en-US" }).model, "english");
  assert.equal(route("I was charged twice this month", { lang: "it" }).model, "multilingual");
  // codes that name no language abstain, so detection decides
  assert.equal(route(it, { lang: "C.UTF-8" }).model, "multilingual");
  assert.equal(route("I was charged twice this month, please refund me today", { lang: "und" }).model, "english");
  const french: unknown = { model: "french" };
  if (untyped<RouteOptions>(french)) assert.throws(() => route(it, french), /unknown model "french"/);
});

test("englishFromCode understands the forms a caller has to hand", () => {
  for (const c of ["en", "EN", "en-US", "en_US.UTF-8", " eng ", "english"]) assert.equal(englishFromCode(c), true, c);
  for (const c of ["it", "pt-BR", "zh_CN.UTF-8", "de"]) assert.equal(englishFromCode(c), false, c);
  for (const c of [undefined, null, "", "  ", "C", "POSIX", "C.UTF-8", "und", "zxx", "mul"]) assert.equal(englishFromCode(c), null, String(c));
});

test("Router.route uses the router's default; a failed load is retried on the next call", async () => {
  const router = new Router({ default: "english", models: { english: { modelDir: "/nonexistent/ped-bundle" } } });
  assert.equal(router.route("Thanks").model, "english");
  assert.equal(router.route("Thanks", { model: "multilingual" }).model, "multilingual");
  const first = router.load("english");
  await assert.rejects(first, /ENOENT/);
  const second = router.load("english");
  assert.notEqual(second, first);
  await assert.rejects(second, /ENOENT/);
  await router.close();
  const klingon: unknown = { default: "klingon" };
  if (untyped<RouterOptions>(klingon)) assert.throws(() => new Router(klingon), /unknown model "klingon"/);
});
