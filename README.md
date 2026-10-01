# PED

<p align="center">
  <img src="https://raw.githubusercontent.com/gabrielegualtieri/ped/main/assets/ped-logo.png" alt="PED — multilingual AI decision model" width="680" />
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

| Checkpoint     | Encoder                | Best at                                                                                                     | Download |
| -------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- | -------- |
| `english`      | ModernBERT-large, 421M | English text                                                                                                | 1.7 GB   |
| `multilingual` | mmBERT-base, 322M      | 100+ languages (Italian, Spanish, German, French, Portuguese, Chinese, Japanese, Arabic, Hindi, Russian, …) | 1.3 GB   |

## Install

```sh
npm install @heryox/ped
```

Node.js 20 or newer. The ONNX weights (fp32) are downloaded from Hugging Face on first use and cached
under `~/.cache/ped` (override with `PED_CACHE`). Budget roughly 2 GB of RAM per loaded checkpoint
plus a few hundred MB per batch of questions.

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
  { subject: "Refund not received", body: "I cancelled two weeks ago and still have no refund..." },
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
result.answers.churn_risk.noul; // 0.0988   (P(true))
result.usage.input_tokens; // 267

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
});
```

Every question of one `systemOne` call is batched into a single run. Runtime depends on the actual
input length, number of questions, execution provider and hardware, rather than `maxLen` alone.

## Exporting the ONNX bundles yourself

`export/export_onnx.py` turns a Hugging Face checkpoint (encoder + decision head) into one ONNX
graph and copies the tokenizer and calibration values next to it. You only need this to build a
bundle from a newer checkpoint or from a variant that is not published:

```sh
cd export
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch transformers safetensors onnx onnxscript onnxruntime huggingface_hub
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('convaiinnovations/laya', local_dir='model', allow_patterns=['model.safetensors','encoder/*','tokenizer/*','rl_agent_config.json','rl_common.py','rl_agent_api.py','multilingual/*'])"
.venv/bin/python export_onnx.py model ../onnx                            # English
.venv/bin/python export_onnx.py model/multilingual ../onnx/multilingual  # multilingual
```

Each run prints the max logit difference vs. PyTorch (≈1e-5). Then `Ped.load({ modelDir: "./onnx" })`,
or publish the bundles with the English one at the repo root and the multilingual one in `multilingual/`:

```sh
hf upload heryox/ped-onnx onnx/multilingual multilingual
```

A bundle is the five files listed in `BUNDLE_FILES`: `ped.onnx`, `ped.onnx.data`, `ped_config.json`,
`tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`.

## Limits

- Each question's options must fit in `head_max_len` tokens (192 English, 256 multilingual);
  `systemOne` throws otherwise. Fewer than about 20 options per `choice` question is the model's own
  recommendation.
- The state is truncated to `max_len` after the question header: 512 tokens for the English checkpoint,
  1024 for the multilingual one (up to 8192 with `maxLen`).
- The multilingual checkpoint ships without fitted temperatures, so its probabilities are the raw
  model's; the English checkpoint's are temperature-calibrated.
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
