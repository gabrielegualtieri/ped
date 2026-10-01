---
license: apache-2.0
base_model:
  - convaiinnovations/laya
  - convaiinnovations/laya-multilingual
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

# Ped — ONNX export of Laya

ONNX export of [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) for use with
[`@heryox/ped`](https://www.npmjs.com/package/@heryox/ped) from Node.js / TypeScript, or with ONNX
Runtime directly. Two checkpoints, one bundle each:

| Folder | Checkpoint | Encoder | Context |
|---|---|---|---|
| repo root | English (`convaiinnovations/laya`) | ModernBERT-large, 421M parameters | 512 |
| `multilingual/` | multilingual, 100+ languages (`convaiinnovations/laya` `multilingual/`) | mmBERT-base, 322M parameters | 1024 (up to 8192) |

Each bundle holds:

| File | Contents |
|---|---|
| `ped.onnx` / `ped.onnx.data` | graph + fp32 weights |
| `ped_config.json` | `max_len`, `head_max_len` and the per-cardinality temperatures from `rl_agent_config.json` |
| `tokenizer/` | the checkpoint's tokenizer |

Inputs: `input_ids` [B,L] int64, `attention_mask` [B,L] int64, `marker_pos` [B,K] int64, `marker_mask` [B,K] bool, `qtype` [B] int64.
Outputs: `logits` [B,K] float32 (uncalibrated; masked slots = -1e4), `act_probs` [B,2] float32.

Built with [`export/export_onnx.py`](https://github.com/gabrielegualtieri/ped/blob/main/export/export_onnx.py);
max logit difference vs. the PyTorch reference ≈ 1e-5.

```ts
import { Router } from "@heryox/ped";
const router = new Router(); // English text -> repo root, other languages -> multilingual/, each downloaded on first use
```

Weights are Convai Innovations' and remain under Apache 2.0. Export code: MIT, https://github.com/gabrielegualtieri/ped
