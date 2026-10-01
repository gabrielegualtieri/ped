/**
 * One ONNX Runtime call over a batch of sequences (rl_common.collate_items): right-pad to the longest
 * sequence and the widest option set, run, and return each sequence's option logits and act probability.
 */
import * as ort from "onnxruntime-node";

export interface Sequence {
  ids: number[];
  markers: number[];
  qtype: number;
}

export async function runSequences(session: ort.InferenceSession, padId: number, seqs: Sequence[]): Promise<{ logits: Float32Array[]; act: number[] }> {
  const n = seqs.length;
  const L = Math.max(...seqs.map((s) => s.ids.length));
  const K = Math.max(...seqs.map((s) => s.markers.length));
  const inputIds = new BigInt64Array(n * L).fill(BigInt(padId));
  const attention = new BigInt64Array(n * L);
  const markerPos = new BigInt64Array(n * K);
  const markerMask = new Uint8Array(n * K);
  const qtype = new BigInt64Array(n);
  seqs.forEach((s, i) => {
    s.ids.forEach((v, j) => {
      inputIds[i * L + j] = BigInt(v);
      attention[i * L + j] = 1n;
    });
    s.markers.forEach((m, j) => {
      markerPos[i * K + j] = BigInt(m);
      markerMask[i * K + j] = 1;
    });
    qtype[i] = BigInt(s.qtype);
  });
  const out = await session.run({
    input_ids: new ort.Tensor("int64", inputIds, [n, L]),
    attention_mask: new ort.Tensor("int64", attention, [n, L]),
    marker_pos: new ort.Tensor("int64", markerPos, [n, K]),
    marker_mask: new ort.Tensor("bool", markerMask, [n, K]),
    qtype: new ort.Tensor("int64", qtype, [n]),
  });
  const logits = out["logits"]?.data;
  const act = out["act_probs"];
  if (!(logits instanceof Float32Array) || !act || !(act.data instanceof Float32Array)) {
    throw new Error("unexpected model outputs (expected float32 logits and act_probs)");
  }
  const actData = act.data;
  const nAct = act.dims[1] ?? 1;
  return {
    logits: seqs.map((s, i) => logits.slice(i * K, i * K + s.markers.length)),
    act: seqs.map((_, i) => actData[i * nAct] ?? 0),
  };
}
