---
license: apache-2.0
base_model:
  - convaiinnovations/laya
library_name: onnx
tags:
  - onnx
  - onnxruntime
  - decision-model
  - system-one
  - jev
  - modernbert
  - mmbert
  - multilingual
---

# PED

<p align="center">
  <img src="https://huggingface.co/heryox/ped-onnx/resolve/main/assets/ped-logo-orbitale.png" alt="PED — multilingual AI decision model" width="680" />
</p>

<p align="center"><strong>Typed decisions. Local inference. Multilingual by design.</strong></p>

<p align="center">
  <a href="https://github.com/gabrielegualtieri/ped">GitHub · source and SDK</a> ·
  <a href="https://www.npmjs.com/package/@heryox/ped">npm · @heryox/ped</a>
</p>

**PED is a multilingual AI decision model and TypeScript toolkit by
[Gabriele Gualtieri (heryox)](https://github.com/gabrielegualtieri).** Turn text, emails, tickets and
structured data into typed choices, ordered scores and yes/no probabilities. Define the decisions
your application needs, then evaluate every question in **one ONNX forward pass**.

This repository publishes PED's English and multilingual ONNX bundles for local inference with
[`@heryox/ped`](https://www.npmjs.com/package/@heryox/ped) or ONNX Runtime directly. The PED project
brings the checkpoints, a strongly typed Node.js SDK and automatic language routing together.

## Built for decisions you can use in code

| Decision | What you define                      | What PED returns                                       |
| -------- | ------------------------------------ | ------------------------------------------------------ |
| `choice` | Named options and their descriptions | Selected option, probability per option and confidence |
| `score`  | An ordered rubric                    | Expected level, distribution and confidence            |
| `noul`   | A yes/no statement                   | P(true)                                                |

- **Local inference:** after the initial download, model inference runs on your machine with ONNX Runtime.
- **English + 100+ languages:** two checkpoints, with automatic or explicit routing through the SDK.
- **Multiple questions, one run:** classification, scoring and binary decisions can share a batch.
- **Native Node.js / TypeScript:** use the SDK without Python or PyTorch at runtime.
- **Task definitions in your application:** describe the options or rubric for each request.

Applications include ticket triage, intent classification, email routing, rubric scoring and
decision steps in automation or agent workflows. Validate the checkpoint on your own task and
language before choosing operating thresholds.

## Quick start

Requires Node.js 20 or newer.

```sh
npm install @heryox/ped
```

```ts
import { Router } from "@heryox/ped";

const router = new Router();

try {
  const result = await router.systemOne(
    "Il pacco doveva arrivare la settimana scorsa e il tracciamento non si aggiorna.",
    {
      department: {
        type: "choice",
        instructions: "Which team should handle this message?",
        criteria: {
          billing: "charges, payments, refunds, invoices",
          shipping: "deliveries, packages, tracking",
          technical: "app bugs, crashes, errors",
        },
      },
      urgent: { type: "noul", instructions: "Does this message need urgent attention?" },
    },
    { lang: "it" },
  );

  console.log(result.answers.department.choice);
  console.log(result.answers.department.probabilities);
  console.log(result.answers.urgent.noul);
  console.log(result.routing.model); // "multilingual"
} finally {
  await router.close();
}
```

Each checkpoint downloads on first use and is cached under `~/.cache/ped` by default. Set
`PED_CACHE` to choose another cache location. Pass `lang` when the language is known, or set
`model: "english"` / `model: "multilingual"` to select a checkpoint directly.

See the [SDK README](https://github.com/gabrielegualtieri/ped#readme) for scoring examples, local
bundles, routing details and runtime options.

## Published checkpoints

| Variant      | Location        | Encoder          | Parameters | Default context                      | Weights       | int8 bundle                    |
| ------------ | --------------- | ---------------- | ---------- | ------------------------------------ | ------------- | ------------------------------ |
| English      | Repository root | ModernBERT-large | ~421M      | 512 tokens                           | ~1.7 GB, fp32 | `int8/`, ~0.68 GB              |
| Multilingual | `multilingual/` | mmBERT-base      | ~322M      | 1024 tokens; configurable up to 8192 | ~1.3 GB, fp32 | `multilingual/int8/`, ~0.40 GB |

The context budget includes the question header, options and state. The SDK truncates the state to
fit, or reads it in windows with `systemOneLong`. Budget roughly 2 GB of RAM per loaded fp32
checkpoint, plus memory for tokenization and batches. Actual memory use and latency depend on inputs,
batch size and hardware.

Each checkpoint also has an int8 bundle (`precision: "int8"` in the SDK): the embedding table and the
weights of the matrix multiplications quantized to int8 per output channel, activations quantized at run
time. The MLP down projections stay in fp32: their inputs carry outlier activations, and quantizing them
cost the English checkpoint 23 points of accuracy. It is 30–40% of the size and about 2x faster on
CPU, within a point of fp32 on MASSIVE in 16 languages; fp32 is the default.

## Bundle layout

Each checkpoint contains the same five files:

| File                              | Contents                                          |
| --------------------------------- | ------------------------------------------------- |
| `ped.onnx`                        | ONNX graph                                        |
| `ped.onnx.data`                   | External weights (fp32, or int8 in `int8/`)       |
| `ped_config.json`                 | Context limits and temperature calibration values |
| `tokenizer/tokenizer.json`        | Tokenizer vocabulary and rules                    |
| `tokenizer/tokenizer_config.json` | Tokenizer configuration                           |

### ONNX interface

| Input            | Type  | Shape    |
| ---------------- | ----- | -------- |
| `input_ids`      | int64 | `[B, L]` |
| `attention_mask` | int64 | `[B, L]` |
| `marker_pos`     | int64 | `[B, K]` |
| `marker_mask`    | bool  | `[B, K]` |
| `qtype`          | int64 | `[B]`    |

| Output      | Type    | Shape    | Meaning                                             |
| ----------- | ------- | -------- | --------------------------------------------------- |
| `logits`    | float32 | `[B, K]` | Uncalibrated option logits; masked slots use `-1e4` |
| `act_probs` | float32 | `[B, 2]` | Action-head probabilities                           |

`B` is the number of questions, `L` the padded sequence length and `K` the padded option count.
The SDK builds the question/state sequences, batches them, applies the configuration's
temperatures and converts outputs into typed answers. When using ONNX Runtime directly,
reproduce this preprocessing and postprocessing; the graph takes tensors rather than raw text.

Since 0.3 the graphs are exported with transformers' `eager` attention instead of `sdpa`: the same
outputs (max logit difference ~1e-5) and the same weights file, faster on CPU.

## Calibration and limits

- **English probabilities:** fitted temperature calibration, including per-option-count values, is supplied in `ped_config.json`.
- **Multilingual probabilities:** all temperatures are `1.0`, with no fitted per-option-count values; these outputs are uncalibrated.
- **Temperature range:** the SDK clamps temperatures to [0.5, 5] at load, as upstream does, so the English `choice:11+` value 0.1006 is applied as 0.5.
- **Your own calibration:** `fitTemperatures` in the SDK fits temperatures per question type and option count on a few hundred labeled examples.
- **Yes/no questions:** the noul head answers "no" to most true statements, so the SDK asks a yes/no question as a two-option choice by default (upstream issue #156); `systemOneLong` keeps the noul head, whose P(true) stays near 0 on unrelated text.
- **Question header budget:** options must fit within 192 tokens for English or 256 for multilingual. The SDK throws if they do not fit. Keep choice sets concise.
- **Context:** long states are truncated after the header and options. Increase `maxLen` for the multilingual checkpoint when needed, up to its supported 8192-token context.
- **Routing:** language detection is heuristic. Explicit `lang` or `model` is preferable when that information is available.
- **Quality:** support for a language does not imply equal accuracy across languages or tasks. Check decisions and probability thresholds against representative data.

## Brand assets

<p>
  <img src="https://huggingface.co/heryox/ped-onnx/resolve/main/assets/ped-icon-orbitale.png" alt="PED standalone orbital symbol" width="128" />
</p>

Download the [PED logo](https://huggingface.co/heryox/ped-onnx/resolve/main/assets/ped-logo-orbitale.png)
or the [standalone symbol](https://huggingface.co/heryox/ped-onnx/resolve/main/assets/ped-icon-orbitale.png)
(square PNG).

## Provenance and licenses

PED is developed and maintained by **Gabriele Gualtieri (heryox)**. The
[SDK and export tooling](https://github.com/gabrielegualtieri/ped) are published under **MIT**, with
the existing upstream notices preserved.

The pretrained encoder and decision-head weights originate from
[Convai Innovations' Laya](https://huggingface.co/convaiinnovations/laya), with the multilingual
checkpoint supplied in that repository's `multilingual/` directory. They remain under
**Apache 2.0**. This release packages those checkpoints for ONNX and adds the PED TypeScript SDK,
batched inference and language routing.

The language detection and routing decision are derived from the upstream `laya` package and
retain its [Apache 2.0 license](https://github.com/gabrielegualtieri/ped/blob/main/licenses/laya-LICENSE).
The request/response format follows the upstream `RLAgent.system_one` reference and TypeSafe Jev's
`system_one` API. The conversion is implemented in
[`export/export_onnx.py`](https://github.com/gabrielegualtieri/ped/blob/main/export/export_onnx.py),
which includes a parity check against the PyTorch reference.
