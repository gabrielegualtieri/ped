# PED

<p align="center">
  <img src="https://raw.githubusercontent.com/gabrielegualtieri/ped/main/assets/ped-logo-orbitale.png" alt="PED — multilingual AI decision model" width="680" />
</p>

<p align="center"><strong>Typed decisions. Local inference. Multilingual by design.</strong></p>

<p align="center">
  <a href="https://www.npmjs.com/package/@heryox/ped">npm · @heryox/ped</a> ·
  <a href="https://huggingface.co/heryox/ped-onnx">Hugging Face · model weights</a> ·
  <a href="https://github.com/gabrielegualtieri/ped">GitHub · source</a>
</p>

**PED is a multilingual AI decision model and TypeScript toolkit by
[Gabriele Gualtieri (heryox)](https://github.com/gabrielegualtieri).** It turns text, tickets, emails
and structured data into decisions your application can use directly: a selected category, an
ordered score or a probability for a yes/no question.

Define the questions and the available answers. PED evaluates them together in **one ONNX forward
pass**, returning structured, typed results. The project combines published ONNX checkpoints, a
Node.js SDK and automatic language routing in one entry point.

## Why PED

- **Application-ready results:** choices, scores and probabilities with types inferred from your questions.
- **Local inference:** run with [ONNX Runtime](https://onnxruntime.ai/) inside Node.js; once the weights are cached, inference stays on your machine.
- **English + 100+ languages:** an English checkpoint and a multilingual checkpoint, selected automatically or explicitly.
- **Multiple questions, one run:** route a ticket, assess its urgency and check a risk signal in the same batch.
- **A native JavaScript workflow:** Node.js / TypeScript at runtime, with the Jev-compatible `systemOne` API.

Use PED for ticket triage, intent classification, email routing, rubric scoring and decision steps
in automation or agent workflows. Your questions define the task; the model returns values that
your code can act on.

## Three decision types

| Type     | What you define                      | What PED returns                                       |
| -------- | ------------------------------------ | ------------------------------------------------------ |
| `choice` | Named options and their descriptions | Selected option, probability per option and confidence |
| `score`  | An ordered rubric                    | Expected level, distribution and confidence            |
| `noul`   | A yes/no statement                   | P(true)                                                |

The English checkpoint applies fitted temperature calibration. The multilingual checkpoint ships
with unit temperatures, so its probabilities are uncalibrated. See [Limits](#limits) for details.

## Two checkpoints, one interface

Two checkpoints are available, and the `Router` picks one per request:

| Checkpoint     | Encoder                | Best at                                                                                                     | Download             |
| -------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------- |
| `english`      | ModernBERT-large, 421M | English text                                                                                                | 1.7 GB, int8 0.68 GB |
| `multilingual` | mmBERT-base, 322M      | 100+ languages (Italian, Spanish, German, French, Portuguese, Chinese, Japanese, Arabic, Hindi, Russian, …) | 1.3 GB, int8 0.40 GB |

## Install

```sh
npm install @heryox/ped
```

Node.js 20 or newer. The ONNX weights are downloaded from Hugging Face on first use and cached under
`~/.cache/ped` (override with `PED_CACHE`): fp32 by default, or the int8 bundles with
`precision: "int8"`, 30–40% of the download and about twice as fast at nearly the same accuracy (see
[Benchmarks](#benchmarks)). Budget roughly 2 GB of RAM per loaded fp32 checkpoint (half to two thirds
of that with int8), plus a few hundred MB per batch of questions.

## Usage: any language

```ts
import { Router } from "@heryox/ped";

const router = new Router();

const questions = {
  department: {
    type: "choice",
    instructions: "Which team should handle this message?",
    criteria: { billing: "charges, payments, refunds, invoices", shipping: "deliveries, packages, tracking", technical: "app bugs, crashes, errors" },
  },
  angry: { type: "noul", instructions: "Is the customer angry?" },
} as const;

const r = await router.systemOne("Il pacco doveva arrivare la settimana scorsa e il tracciamento non si aggiorna da giorni.", questions);

r.answers.department.choice; // "shipping"
r.answers.department.probabilities; // { billing: 0.0001, shipping: 0.9898, technical: 0.0101 }
r.routing.model; // "multilingual"
r.routing.reason; // "Latin script, language looks like it, not English"

await router.close();
```

English text goes to the English checkpoint and everything else to the multilingual one. A
checkpoint is downloaded and loaded the first time a request needs it, so an English-only service
never downloads the multilingual weights, and the other way round.

### How the router decides

In order: an explicit `model`, then an explicit `lang`, then detection, then the default.

```ts
await router.systemOne(state, questions, { model: "multilingual" }); // skip detection
await router.systemOne(state, questions, { lang: "it" }); // you know the language: "it", "en-US", "pt_BR.UTF-8"
router.route(state); // { model, reason, detection } without loading or running anything
```

Detection uses dependency-free TypeScript heuristics:

- a non-Latin script (Chinese, Japanese, Korean, Arabic, Cyrillic, Devanagari, Greek, Hebrew, Thai, …)
  goes to `multilingual`: the English checkpoint cannot read it and stays confident while wrong;
- Latin text goes to `multilingual` when its function words or its accented letters say it is not
  English (`il`, `della`, `perché`, `você`, `für`, …);
- in a structured state or a multi-line text, one non-English field or line is enough: a Portuguese
  message with an English stack trace goes to `multilingual`;
- a state with no clue — no letters, or a short Latin text without accents such as "Grazie mille" or
  "Thanks" — goes to the default, `multilingual` unless you set `new Router({ default: "english" })`.
  English read by the multilingual checkpoint loses a little accuracy; another language read by the
  English checkpoint gets confidently wrong answers.

The two checkpoints have different language strengths. Pass `lang` or `model` when the language is
known; automatic routing handles the remaining cases and includes its reason in the result.

### Router options

```ts
new Router({
  default: "multilingual", // checkpoint for states that give no clue
  models: {
    // per-checkpoint options, over the shared ones below
    english: { modelDir: "./onnx" },
    multilingual: { modelDir: "./onnx/multilingual", maxLen: 8192 },
  },
  // ...and every Ped.load option (cacheDir, revision, token, onProgress, executionProviders, ...)
});

await router.preload(); // load both checkpoints now instead of on first use
```

## Usage: one checkpoint

`Ped` runs a single checkpoint; `Router` is built on it.

```ts
import { Ped } from "@heryox/ped";

const ped = await Ped.load(); // English; Ped.load({ subfolder: "multilingual" }) for the multilingual one

const result = await ped.systemOne(
  { subject: "Refund not received", body: "I cancelled two weeks ago and still have no refund... I will dispute the charge with my bank." },
  {
    department: {
      type: "choice",
      instructions: "Which team should handle this ticket?",
      criteria: { billing: "payments, refunds, invoices", support: "product help and bugs", sales: "new purchases" },
    },
    urgency: {
      type: "score",
      instructions: "How urgent is this ticket?",
      criteria: ["not urgent", "somewhat urgent", "urgent", "critical"],
    },
    churn_risk: { type: "noul", instructions: "Is the customer likely to cancel or dispute?" },
  },
);

result.answers.department.choice; // "billing"
result.answers.department.probabilities; // { billing: 0.9415, support: 0.031, sales: 0.0275 }
result.answers.urgency.score; // 1.3886   (expected level, 0..3)
result.answers.churn_risk.noul; // 0.8669   (P(true))
result.answers.churn_risk.answer_confidence; // 0.8669   (the probability of the reported answer)
result.usage.input_tokens; // 256

await ped.close();
```

The answer types follow the question types, so `result.answers.department` is a `ChoiceAnswer`
and `result.answers.churn_risk` a `NoulAnswer` without any casting.

### Options

```ts
await Ped.load({
  modelDir: "./onnx", // use a local export instead of downloading (see below)
  repo: "heryox/ped-onnx", // Hugging Face repo that holds the ONNX bundle
  subfolder: "multilingual", // a checkpoint variant inside that repo (none = English)
  revision: "main", // pin a commit hash for reproducible results; "main" follows the repo
  cacheDir: "/var/cache/ped",
  token: process.env.HF_TOKEN, // for private repos
  onProgress: ({ file, received, total }) => {}, // download progress
  executionProviders: ["cpu"], // onnxruntime-node execution providers
  sessionOptions: { intraOpNumThreads: 4 },
  maxLen: 8192, // longer states: the multilingual checkpoint reads up to 8192 tokens (default 1024)
  precision: "int8", // the int8 bundle: 30-40% of the download, ~2x faster, about as accurate (Benchmarks)
  noul: "choice", // yes/no as a two-option choice (systemOne's default); "native": the noul head (systemOneLong's)
  optionOrders: 1, // >1 averages each choice over that many option orders (slower, removes position bias)
  temperatures: { temperature_by_options: { "choice:3-5": 1.4 } }, // e.g. from fitTemperatures, clamped to [0.5, 5]
});
```

Every question of one `systemOne` call is batched into a single run. Runtime depends on the actual
input length, number of questions, execution provider and hardware, rather than `maxLen` alone.

### What every answer and result carries

- `answer_confidence` on every answer: the probability of the reported option (max p), the number
  to put a threshold on once the probabilities are calibrated for your task. `confidence` is
  1 − normalized entropy: how concentrated the whole distribution is.
- `usage.truncated` / `usage.state_tokens_dropped` / `usage.truncated_questions`: whether the state
  had to be cut to fit the sequence, and by how much. A truncated state was not fully read: use
  `systemOneLong`.
- `usage.options` (only when it happens): a question had so many or such long options that some lost
  their own tokens to the 192/256-token header budget.

## Long documents

`systemOne` reads at most `max_len` tokens (512 English, 1024 multilingual) and cuts the rest.
`systemOneLong` reads the whole state in overlapping windows instead and combines the answers per
question: a yes/no takes the window with the highest P(true) (the statement holds if any part of the
document supports it), a choice or a score takes the most confident window. `answer.window` says
which window decided, as token offsets into the state.

```ts
const r = await router.systemOneLong(longEmailThread, {
  refund: { type: "noul", instructions: "Does the customer ask for a refund?" },
  topic: { type: "choice", instructions: "What is the thread about?", criteria: ["billing", "shipping", "technical"] },
});
r.answers.refund.noul; // decided by the window that holds the request
r.answers.refund.window; // { index: 4, token_start: 624, token_end: 936, count: 6 }
r.usage.windows; // 6
```

Yes/no questions are read with the checkpoint's own noul head here (`noul: "native"`), not as the
two-option choice `systemOne` uses by default: taking the highest P(true) over many windows needs a
P(true) that stays near 0 on unrelated text, and the two-option choice reached 0.99 on some unrelated
windows of a test document. Set `noul` to override. A state that fits one window gets exactly
`systemOne`'s answers with the same `noul`. Options: `window` (state tokens per window), `stride`
(default half a window). A conversation passed as a list keeps its newest turns when `systemOne` has to
cut it.

## Many states at once

```ts
const results = await router.systemOneBatch(tickets, questions); // one result per ticket, same order
```

States of similar length share ONNX Runtime runs (`batchSize`, default 64 sequences per run). On the
4-core benchmark machine that is 5–15% faster than one `systemOne` per state, since ONNX Runtime already
spreads a single request over the cores. The router routes every state on its own.

## Calibrating probabilities on your data

The checkpoints are trained to report honest probabilities, but they come out over-confident on new
tasks. A few hundred labeled examples are enough to fit one temperature per question type and option
count:

```ts
const fit = await ped.fitTemperatures(examples.map((e) => ({ state: e.text, questions, labels: { department: e.team, urgent: e.isUrgent } })));
fit.after.ece; // expected calibration error on the examples, before: fit.before.ece
ped.setTemperatures(fit); // or Ped.load({ temperatures: fit }) next time
```

Labels are the option key for a choice, the level index for a score and `true` / `false` for a yes/no.
Temperatures are clamped to [0.5, 5] (as upstream does); buckets with fewer than `minPerBucket` (100)
examples fall back to the question type's temperature.

## Benchmarks

Every number here comes from the scripts in [`bench/`](bench/README.md) and the result files in
`bench/results/` (`yarn bench:report` prints these tables), through the public API, with each
version's published bundles. Accuracy does not depend on the machine; latency is the one named below.

<!-- bench:start -->

Measured on a 4-core cloud VM (Intel Xeon @ 2.10 GHz, AVX-512 VNNI), Node 22, ONNX Runtime 1.30 on CPU.

### Accuracy and calibration: MASSIVE, 16 languages

[MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_scenario) voice-assistant requests, the same
100 in each of 16 languages (en, it, es, fr, de, pt, ru, zh, ja, ar, hi, ko, tr, pl, th, sw). Each request
is one `router.systemOne` call with three questions: its scenario among 18 (a choice), and two yes/no
questions, one true and one false. The router picks the checkpoint as in production; latency is per call.

| Version    | Scenario accuracy | Scenario ECE | Yes/no accuracy | Yes/no AUROC | p50 per call | p95 per call |
| ---------- | ----------------- | ------------ | --------------- | ------------ | ------------ | ------------ |
| 0.2.0      | 56.2%             | 0.187        | 60.0%           | 0.776        | 651 ms       | 1,383 ms     |
| 0.3.0      | 56.2%             | 0.177        | 70.2%           | 0.799        | 562 ms       | 1,194 ms     |
| 0.3.0 int8 | 56.1%             | 0.177        | 69.4%           | 0.793        | 322 ms       | 604 ms       |

<details><summary>Per language</summary>

| Language | Scenario 0.2.0 | Scenario 0.3.0 | Scenario 0.3.0 int8 | Yes/no 0.2.0 | Yes/no 0.3.0 | Yes/no 0.3.0 int8 |
| -------- | -------------- | -------------- | ------------------- | ------------ | ------------ | ----------------- |
| en       | 75.0%          | 75.0%          | 72.0%               | 66.0%        | 78.0%        | 78.0%             |
| it       | 58.0%          | 58.0%          | 58.0%               | 55.0%        | 66.0%        | 63.5%             |
| es       | 58.0%          | 58.0%          | 59.0%               | 59.5%        | 72.0%        | 71.0%             |
| fr       | 63.0%          | 63.0%          | 66.0%               | 61.0%        | 69.0%        | 68.0%             |
| de       | 62.0%          | 62.0%          | 60.0%               | 59.5%        | 69.0%        | 69.5%             |
| pt       | 59.0%          | 59.0%          | 61.0%               | 54.5%        | 71.0%        | 68.0%             |
| ru       | 63.0%          | 63.0%          | 63.0%               | 63.0%        | 72.0%        | 72.5%             |
| zh       | 59.0%          | 59.0%          | 55.0%               | 60.0%        | 69.0%        | 68.0%             |
| ja       | 66.0%          | 66.0%          | 62.0%               | 59.0%        | 70.5%        | 73.0%             |
| ar       | 50.0%          | 50.0%          | 50.0%               | 60.0%        | 65.0%        | 64.0%             |
| hi       | 53.0%          | 53.0%          | 57.0%               | 61.5%        | 74.5%        | 74.5%             |
| ko       | 56.0%          | 56.0%          | 55.0%               | 60.0%        | 73.5%        | 75.0%             |
| tr       | 50.0%          | 50.0%          | 53.0%               | 61.5%        | 76.0%        | 76.0%             |
| pl       | 52.0%          | 52.0%          | 54.0%               | 61.5%        | 68.5%        | 67.0%             |
| th       | 58.0%          | 58.0%          | 54.0%               | 59.5%        | 69.0%        | 67.0%             |
| sw       | 17.0%          | 17.0%          | 19.0%               | 59.0%        | 60.0%        | 55.5%             |

</details>

- **Yes/no:** 0.3.0 answers 70% of the yes/no questions right instead of 60%, better in all 16
  languages, and ranks true above false statements more often (AUROC 0.776 → 0.799).
- **Scenario:** the same answers, since the model is the same. On the English requests the clamped
  `choice:11+` temperature takes the calibration error (ECE) from 0.186 to 0.118.
- **int8:** within a point of fp32 overall (single languages move by up to 4 points either way, as
  100 requests allow), 1.7x faster per call.
- Swahili is the multilingual checkpoint's weak spot (17%, against 6% by chance).

### Speed

Median of 15 runs per case, one process per configuration. Short cases vary by about ±10% between runs
on this VM. The `systemOneBatch` rows are 32 tickets in one call; against 32 `systemOne` calls in the same
process they are 5–15% faster.

**English checkpoint** (p50 of each case)

|                                                             | 0.2.0    | 0.3.0     | 0.3.0 int8 |
| ----------------------------------------------------------- | -------- | --------- | ---------- |
| Load                                                        | 2,532 ms | 2,407 ms  | 1,536 ms   |
| Memory after load (RSS)                                     | 1,662 MB | 1,660 MB  | 808 MB     |
| 1 question, short ticket                                    | 221 ms   | 162 ms    | 86 ms      |
| 3 questions, short ticket                                   | 543 ms   | 518 ms    | 208 ms     |
| 10 questions, short ticket                                  | 1,655 ms | 1,510 ms  | 650 ms     |
| 1 question, ~450-token message                              | 1,526 ms | 1,093 ms  | 609 ms     |
| 3 questions, 32 short tickets in one systemOneBatch (total) | –        | 14,742 ms | 6,687 ms   |

**Multilingual checkpoint** (p50 of each case)

|                                                             | 0.2.0    | 0.3.0    | 0.3.0 int8 |
| ----------------------------------------------------------- | -------- | -------- | ---------- |
| Load                                                        | 2,092 ms | 2,034 ms | 1,961 ms   |
| Memory after load (RSS)                                     | 888 MB   | 886 MB   | 587 MB     |
| 1 question, short ticket                                    | 80 ms    | 67 ms    | 35 ms      |
| 3 questions, short ticket                                   | 195 ms   | 187 ms   | 100 ms     |
| 10 questions, short ticket                                  | 650 ms   | 568 ms   | 306 ms     |
| 1 question, ~450-token message                              | 607 ms   | 457 ms   | 299 ms     |
| 1 question, ~2,000-token document                           | 7,203 ms | 4,077 ms | 3,673 ms   |
| 3 questions, 32 short tickets in one systemOneBatch (total) | –        | 6,313 ms | 2,988 ms   |
| 1 question, ~2,000-token document, systemOneLong            | –        | 5,323 ms | 3,434 ms   |

<!-- bench:end -->

Coming next: the same requests against Jev, accuracy and latency side by side, and the full MASSIVE
test set (51 languages, 2,974 requests each), on a 64-core server.

## Exporting the ONNX bundles yourself

`export/export_onnx.py` turns a Hugging Face checkpoint (encoder + decision head) into one ONNX
graph and copies the tokenizer and calibration values next to it. You only need this to build a
bundle from a newer checkpoint or from a variant that is not published:

```sh
cd export
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('convaiinnovations/laya', local_dir='model', allow_patterns=['model.safetensors','encoder/*','tokenizer/*','rl_agent_config.json','rl_common.py','rl_agent_api.py','multilingual/*'])"
.venv/bin/python export_onnx.py model ../onnx --int8                            # English (+ int8/)
.venv/bin/python export_onnx.py model/multilingual ../onnx/multilingual --int8  # multilingual (+ int8/)
```

Each run prints the max logit difference against the PyTorch reference (~1e-5) and, with `--int8`, the
int8 bundle's difference from the fp32 one. The graph uses transformers' `eager` attention: the same
results as the `sdpa` graph of 0.2 without its NaN guards around every attention layer, and faster at
every length measured (`--attention sdpa` exports the 0.2 graph). `--attention banded`
(`export/ped_attention.py`) computes the sliding-window layers as a band, each token against the 128
around it, instead of a full L × L matrix: twice as fast as the 0.2 graph on 2,000–4,000-token inputs with
a third less memory, but slower below about 1,000 tokens, so the published bundles do not use it. Then
`Ped.load({ modelDir: "./onnx" })`, or publish the bundles with the English one at the repo root, the
multilingual one in `multilingual/` and each int8 bundle in its `int8/` subfolder:

```sh
hf upload heryox/ped-onnx onnx . --exclude "multilingual/*"  # English bundle, its int8/ and the model card
hf upload heryox/ped-onnx onnx/multilingual multilingual     # multilingual bundle and its int8/
```

`export_onnx.py --quantize ../onnx` builds `../onnx/int8` from an fp32 bundle you already have, without
PyTorch. The export needs about 5.5 GB of memory for the English checkpoint; the quantization runs in its
own process and peaks at about 7.5 GB.

A bundle is the five files listed in `BUNDLE_FILES`: `ped.onnx`, `ped.onnx.data`, `ped_config.json`,
`tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`.

## Limits

- Each question's options must fit in `head_max_len` tokens (192 English, 256 multilingual);
  `systemOne` throws otherwise. Fewer than about 20 options per `choice` question is the model's own
  recommendation.
- The state is truncated to `max_len` after the question header: 512 tokens for the English checkpoint,
  1024 for the multilingual one (up to 8192 with `maxLen`). `systemOneLong` reads longer states in
  windows.
- The multilingual checkpoint ships without fitted temperatures, so its probabilities are the raw
  model's, and over-confident: on MASSIVE the best temperature is between 1.4 and 2.4 in every language
  measured. Fit your own with `fitTemperatures`. The English checkpoint's are temperature-calibrated.
- Language detection is a heuristic: script detection is exact, the Latin-language guess is
  best-effort. Pass `lang` or `model` when you already know.
- A JSON state is serialized like Python's `json.dumps(ensure_ascii=False)` so that tokens match the
  reference implementation; non-integer numbers may format differently between JS and Python.

## Development

```sh
yarn install
yarn test        # unit tests; the model tests run when ./onnx and ./onnx/multilingual hold bundles (or PED_MODEL_DIR / PED_MODEL_DIR_MULTILINGUAL)
yarn typecheck
yarn build
PED_MODEL_DIR=./onnx yarn example
yarn example:router
```

## Brand assets

<p>
  <img src="https://raw.githubusercontent.com/gabrielegualtieri/ped/main/assets/ped-icon-orbitale.png" alt="PED standalone orbital symbol" width="128" />
</p>

Download the [PED logo](https://raw.githubusercontent.com/gabrielegualtieri/ped/main/assets/ped-logo-orbitale.png)
or the [standalone symbol](https://raw.githubusercontent.com/gabrielegualtieri/ped/main/assets/ped-icon-orbitale.png)
(square PNG).

## Credits and licenses

PED is developed and maintained by **Gabriele Gualtieri (heryox)**. The PED SDK and export tooling
are published under [MIT](LICENSE), with the existing upstream notices preserved.

The pretrained encoder and decision-head weights originate from
[Convai Innovations' Laya](https://huggingface.co/convaiinnovations/laya), including its
`multilingual/` checkpoint, and remain under **Apache 2.0**. This release packages those checkpoints
for ONNX and adds the PED TypeScript SDK, batched inference and language routing. The request and
response format follows the upstream `RLAgent.system_one` reference and TypeSafe Jev's `system_one`
API.

The language detection and routing decision (`src/lang*.ts`, `src/router.ts`) are ported from the
upstream `laya` package and remain under Apache 2.0; see [the retained license](licenses/laya-LICENSE).
