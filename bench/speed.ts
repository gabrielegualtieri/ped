/**
 * Latency, load time and memory of one checkpoint through the public API. Run one process per
 * configuration, so the resident memory it reports belongs to that checkpoint alone.
 *
 *   npx tsx bench/speed.ts [--dir DIR | --options '{"subfolder":"multilingual","precision":"int8"}'] [--long] [--module path]
 *
 * Cases: 1, 3 and 10 questions about a short ticket, one question about a ~450-token message, and with
 * --long one question about a ~2,000-token document (`maxLen: 4096`, multilingual only). Builds that have
 * them (0.3+) also time `systemOneBatch` on 32 tickets and, with --long, `systemOneLong` on the document in
 * the default windows of a 1,024-token sequence.
 */
import { readFileSync } from "node:fs";
import { cpus } from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import { pathToFileURL } from "node:url";

const { values: args } = parseArgs({
  options: {
    dir: { type: "string" },
    options: { type: "string", default: "{}" },
    long: { type: "boolean", default: false },
    runs: { type: "string", default: "7" },
    module: { type: "string", default: path.resolve(import.meta.dirname, "../src/index.ts") },
  },
});

interface Result {
  usage: { input_tokens: number };
}
interface PedLike {
  systemOne(state: unknown, questions: object): Promise<Result>;
  systemOneBatch?(states: unknown[], questions: object): Promise<Result[]>;
  systemOneLong?(state: unknown, questions: object, opts: object): Promise<Result>;
  close(): Promise<void>;
}
const lib = (await import(pathToFileURL(path.resolve(args.module)).href)) as { Ped: { load(opts: object): Promise<PedLike> } };
const version = (JSON.parse(readFileSync(path.resolve(path.dirname(args.module), "../package.json"), "utf8")) as { version: string }).version;
const mb = () => Math.round(process.memoryUsage().rss / 1e6);

const t0 = performance.now();
const ped = await lib.Ped.load({
  ...(JSON.parse(args.options) as object),
  ...(args.dir ? { modelDir: args.dir } : {}),
  ...(args.long ? { maxLen: 4096 } : {}),
});
const loadMs = performance.now() - t0;
const rssLoaded = mb();

const ticket = { subject: "Refund not received", body: "I cancelled my subscription two weeks ago and I still have not received my refund." };
const q3 = {
  department: {
    type: "choice",
    instructions: "Which team should handle this ticket?",
    criteria: { billing: "payments, refunds", support: "product help", sales: "new purchases" },
  },
  urgency: { type: "score", instructions: "How urgent is this ticket?", criteria: ["not urgent", "somewhat urgent", "urgent", "critical"] },
  churn: { type: "noul", instructions: "Is the customer likely to cancel or dispute?" },
};
const q1 = { churn: q3.churn };
const q10 = Object.fromEntries(Array.from({ length: 10 }, (_, i) => [`q${i}`, Object.values(q3)[i % 3]]));
const para =
  "Hello, I am writing about order 4521 which I placed three weeks ago. The tracking page has not changed since the parcel left the warehouse, " +
  "and two emails to your support address have gone unanswered. I paid for express delivery because the items were a birthday present, so the delay " +
  "has already cost me more than the shipping fee. ";

async function time(label: string, call: () => Promise<Result | Result[]>, runs: number) {
  for (let i = 0; i < 2; i++) await call();
  const ms: number[] = [];
  let tokens = 0;
  for (let i = 0; i < runs; i++) {
    const s = performance.now();
    const r = await call();
    ms.push(performance.now() - s);
    tokens = [r].flat().reduce((sum, x) => sum + x.usage.input_tokens, 0);
  }
  ms.sort((a, b) => a - b);
  return { case: label, p50_ms: Math.round(ms[Math.floor(ms.length / 2)] ?? 0), input_tokens: tokens };
}
const one = (state: unknown, questions: object) => () => ped.systemOne(state, questions);

const runs = Number(args.runs);
const cases = [
  await time("1 question, short ticket", one(ticket, q1), runs),
  await time("3 questions, short ticket", one(ticket, q3), runs),
  await time("10 questions, short ticket", one(ticket, q10), runs),
  await time("1 question, ~450-token message", one(para.repeat(6), q1), runs),
];
const batch = ped.systemOneBatch?.bind(ped);
const long = ped.systemOneLong?.bind(ped);
if (batch) {
  const tickets = Array.from({ length: 32 }, (_, i) => ({ ...ticket, subject: `${ticket.subject} (#${i + 1})` }));
  cases.push(await time("3 questions, 32 short tickets in one systemOneBatch (total)", () => batch(tickets, q3), 3));
}
if (args.long) cases.push(await time("1 question, ~2,000-token document", one(para.repeat(28), q1), 3));
// in the windows systemOneLong reads with the default 1,024-token sequence: 1024 - 256 (options) - 8 = 760 state tokens
if (args.long && long) cases.push(await time("1 question, ~2,000-token document, systemOneLong", () => long(para.repeat(28), q1, { window: 760 }), 3));
console.log(
  JSON.stringify(
    {
      version,
      options: JSON.parse(args.options) as object,
      dir: args.dir,
      hardware: { cpu: cpus()[0]?.model ?? "?", cores: cpus().length, node: process.version },
      load_ms: Math.round(loadMs),
      rss_mb: rssLoaded,
      cases,
    },
    null,
    1,
  ),
);
await ped.close();
