"""Export the decision model (convaiinnovations/laya: encoder + decision head; English or multilingual checkpoint) to ONNX.

Usage:  .venv/bin/python export_onnx.py [model_dir] [out_dir] [--int8] [--attention eager|banded|sdpa]
        .venv/bin/python export_onnx.py --quantize bundle_dir
        model_dir is a checkpoint (the repo root, or a variant subfolder such as model/multilingual)
        --int8 also writes an int8 bundle to out_dir/int8 (dynamic quantization of the weight matmuls
        and the embedding table, the MLP down projections kept in fp32): 30-40% of the size, ~2x faster
        --quantize writes bundle_dir/int8 from an existing fp32 bundle, without PyTorch
        --attention eager (default) exports transformers' "eager" attention: the same results as "sdpa"
        (max logit difference ~1e-5) without its NaN guards: 4-28% faster up to ~450 tokens, 43% at 2,000.
        "banded" (ped_attention.py) is faster still above ~1,000 tokens (2x at 2,000-4,000) but slower
        below; "sdpa" reproduces the 0.2 graph.
Inputs : input_ids [B,L] int64, attention_mask [B,L] int64, marker_pos [B,K] int64, marker_mask [B,K] bool, qtype [B] int64
Outputs: logits [B,K] float32 (uncalibrated; masked slots = -1e4), act_probs [B,2] float32
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

argv = sys.argv[1:]
attention = argv[argv.index("--attention") + 1] if "--attention" in argv else "eager"
args = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--attention")]
int8 = "--int8" in argv


def bundle_mb(d):
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.startswith("ped")) / 1e6


def quantize(bundle_dir):
    """bundle_dir/int8: int8 weights, one scale per output channel, for the embedding table and every MatMul against a
    constant except the down projections; activations are quantized on the fly."""
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q_dir = os.path.join(bundle_dir, "int8")
    os.makedirs(q_dir, exist_ok=True)
    proto = onnx.load(os.path.join(bundle_dir, "ped.onnx"))
    # the dynamo export records shapes from the trace example; let the quantizer's shape inference redo them
    proto.graph.ClearField("value_info")
    dims = {i.name: list(i.dims) for i in proto.graph.initializer}
    hidden = max((dims[n.input[0]] for n in proto.graph.node if n.op_type == "Gather" and n.input[0] in dims), key=lambda d: d[0])[1]
    # The down projections ([wider, hidden]: every MLP's output) read activations with outliers that one scale per
    # tensor flattens. Quantized, they took the English checkpoint from 71% to 48% on 100 MASSIVE requests and to
    # 57% agreement with fp32; kept in fp32 (a fifth of the matmul weights), 72% and 93%.
    down = [n.name for n in proto.graph.node
            if n.op_type == "MatMul" and n.input[1] in dims and dims[n.input[1]][1] == hidden and dims[n.input[1]][0] > hidden]
    quantize_dynamic(proto, os.path.join(q_dir, "ped.onnx"), op_types_to_quantize=["MatMul", "Gather"], weight_type=QuantType.QInt8,
                     per_channel=True, use_external_data_format=True, nodes_to_exclude=down, extra_options={"MatMulConstBOnly": True})
    del proto
    shutil.copytree(os.path.join(bundle_dir, "tokenizer"), os.path.join(q_dir, "tokenizer"), dirs_exist_ok=True)
    shutil.copy(os.path.join(bundle_dir, "ped_config.json"), q_dir)
    # two rows of random tokens, the second padded, four option markers
    mask = np.ones((2, 40), dtype=np.int64)
    mask[1, 30:] = 0
    feeds = {"input_ids": np.random.default_rng(0).integers(5, 1000, (2, 40)), "attention_mask": mask,
             "marker_pos": np.array([[3, 9, 15, 21], [3, 9, 0, 0]]), "marker_mask": np.array([[True] * 4, [True, True, False, False]]),
             "qtype": np.array([0, 2])}
    run = lambda d: ort.InferenceSession(os.path.join(d, "ped.onnx"), providers=["CPUExecutionProvider"]).run(None, feeds)[0]  # noqa: E731
    fp32, q = run(bundle_dir), run(q_dir)
    print("int8: max |dlogits| vs fp32 =", np.abs(q - fp32).max(), " same argmax:", bool((q.argmax(-1) == fp32.argmax(-1)).all()))
    print("wrote", os.path.join(q_dir, "ped.onnx"), "%.0f MB" % bundle_mb(q_dir))


if "--quantize" in argv:
    quantize(os.path.abspath(args[0]))
    sys.exit()

import torch  # noqa: E402

model_dir = os.path.abspath(args[0] if len(args) > 0 else "model")
out_dir = os.path.abspath(args[1] if len(args) > 1 else "../onnx")

# rl_common.py sits at the repo root; a variant subfolder (multilingual/, typed-decisions/) shares it
sys.path.insert(0, os.path.dirname(model_dir))
sys.path.insert(0, model_dir)
from rl_common import build_model  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ped_attention  # noqa: E402,F401  (registers the "banded" attention implementation)

os.makedirs(out_dir, exist_ok=True)

from safetensors.torch import load_file  # noqa: E402

cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
# the reference ("sdpa", as build_model makes it) stays for the parity check; the export uses `attention`
reference = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
reference.load_state_dict(model.state_dict(), strict=True)
reference.eval()
model.encoder.config._attn_implementation = attention
model.eval()
model.encoder.config.reference_compile = False


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act = self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits, torch.softmax(act.float(), -1)


w = Wrapper(model)
B, L, K = 2, 40, 4
ex = (
    torch.randint(5, 1000, (B, L)),
    torch.ones(B, L, dtype=torch.long),
    torch.tensor([[3, 9, 15, 21], [3, 9, 0, 0]]),
    torch.tensor([[True, True, True, True], [True, True, False, False]]),
    torch.tensor([0, 2]),
)
ex[1][1, 30:] = 0  # second row padded

out = os.path.join(out_dir, "ped.onnx")
# grad must stay enabled: nn.TransformerEncoderLayer's fused "fast path" (not exportable) is only taken under no_grad.
batch, seq, opts = torch.export.Dim("batch"), torch.export.Dim("seq", min=8), torch.export.Dim("options", min=2)
prog = torch.onnx.export(
    w, ex, opset_version=18, dynamo=True, optimize=True,
    input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
    output_names=["logits", "act_probs"],
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq},
                    "marker_pos": {0: batch, 1: opts}, "marker_mask": {0: batch, 1: opts}, "qtype": {0: batch}},
)
prog.save(out, external_data=True)  # ped.onnx (graph) + ped.onnx.data (weights), the two files BUNDLE_FILES expects

# ship the tokenizer + calibration config next to the graph
shutil.copytree(os.path.join(model_dir, "tokenizer"), os.path.join(out_dir, "tokenizer"), dirs_exist_ok=True)
json.dump({k: cfg[k] for k in ("max_len", "head_max_len", "temperature", "temperature_by_options")},
          open(os.path.join(out_dir, "ped_config.json"), "w"), indent=1)

# parity check
import onnxruntime as ort  # noqa: E402

with torch.no_grad():
    ref_logits, ref_act = Wrapper(reference)(*ex)
sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
o = sess.run(None, {"input_ids": ex[0].numpy(), "attention_mask": ex[1].numpy(), "marker_pos": ex[2].numpy(),
                    "marker_mask": ex[3].numpy(), "qtype": ex[4].numpy()})
print("attention=%s: max |dlogits| vs the PyTorch sdpa reference =" % attention, np.abs(o[0] - ref_logits.numpy()).max(),
      " max |dact| =", np.abs(o[1] - ref_act.numpy()).max())
print("wrote", out, "%.0f MB" % bundle_mb(out_dir))

if int8:
    # in a fresh process: the quantizer keeps a few copies of the graph, and memory freed here is not all
    # returned to the system
    del model, reference, w, prog, sess
    sys.stdout.flush()
    subprocess.run([sys.executable, os.path.abspath(__file__), "--quantize", out_dir], check=True)
