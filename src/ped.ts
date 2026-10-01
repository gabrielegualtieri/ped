/**
 * Ped — TypeScript port of the checkpoint's rl_agent_api.py on top of onnxruntime-node.
 *
 * `Ped.load()` resolves the ONNX bundle (a local directory, or the published Hugging Face repo, cached
 * under ~/.cache/ped), builds the tokenizer, and opens the session. `systemOne()` then answers
 * any number of typed questions about one state in a single forward pass, exactly as the Python
 * reference does (same sequence layout, same per-cardinality temperature, same rounding).
 */
import { readFile } from "node:fs/promises";
import path from "node:path";
import * as ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { buildSequence, confidenceFromProbs, QTYPES, renderOptions, softmax, tempBucket, toInternal, type SpecialIds } from "./sequence.js";
import { ensureBundle, type DownloadOptions } from "./download.js";
import { alignWithRustTokenizer, specialIds, type TokenizerConfig, type TokenizerJson } from "./tokenizer.js";
import type { Answer, PedConfig, Question, SystemOneResult } from "./types.js";

export interface PedOptions extends DownloadOptions {
  /**
   * Directory holding ped.onnx, ped.onnx.data, ped_config.json and tokenizer/ (the output of
   * export/export_onnx.py). When given, nothing is downloaded and the Hugging Face options are ignored.
   */
  modelDir?: string;
  /** onnxruntime execution providers (default: ["cpu"]) */
  executionProviders?: ort.InferenceSession.ExecutionProviderConfig[];
  /** extra onnxruntime session options (merged over the defaults) */
  sessionOptions?: ort.InferenceSession.SessionOptions;
  /**
   * Tokens of the request sequence (question header + options + state); the state is cut to fit. Default:
   * the bundle's `max_len` (512 English, 1024 multilingual). The multilingual checkpoint reads up to 8192
   * for long documents; time grows with the real input length, not with the limit.
   */
  maxLen?: number;
}

const round4 = (x: number) => Math.round(x * 1e4) / 1e4;

export class Ped {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly tok: Tokenizer,
    readonly config: PedConfig,
    private readonly ids: SpecialIds,
    /** where the bundle was loaded from */
    readonly modelDir: string,
  ) {}

  static async load(opts: PedOptions = {}): Promise<Ped> {
    const modelDir = opts.modelDir ? path.resolve(opts.modelDir) : await ensureBundle(opts);
    const read = async (f: string): Promise<unknown> => JSON.parse(await readFile(path.join(modelDir, f), "utf8"));
    const config = (await read("ped_config.json")) as PedConfig;
    if (opts.maxLen !== undefined) {
      if (!Number.isInteger(opts.maxLen) || opts.maxLen < 1) throw new Error(`maxLen must be a positive integer, got ${opts.maxLen}`);
      config.max_len = opts.maxLen;
    }
    const tokConfig = (await read("tokenizer/tokenizer_config.json")) as TokenizerConfig;
    const tokJson = (await read("tokenizer/tokenizer.json")) as TokenizerJson;
    const tok = new Tokenizer(tokJson, tokConfig);
    alignWithRustTokenizer(tok, tokJson);
    const ids = specialIds(tok, tokConfig);
    const session = await ort.InferenceSession.create(path.join(modelDir, "ped.onnx"), {
      executionProviders: opts.executionProviders ?? ["cpu"],
      graphOptimizationLevel: "all",
      ...opts.sessionOptions,
    });
    return new Ped(session, tok, config, ids, modelDir);
  }

  private readonly encode = (text: string): number[] => this.tok.encode(text, { add_special_tokens: false }).ids;

  /** Answer every question about `state` in one forward pass (Jev's `system_one` request/response shape). */
  async systemOne<Q extends Record<string, Question>>(state: unknown, questions: Q): Promise<SystemOneResult<Q>> {
    const qids = Object.keys(questions);
    if (qids.length === 0) {
      throw new Error("systemOne: at least one question is required");
    }
    const items = qids.map((qid) => {
      const q = toInternal(questions[qid] as Question);
      const { ids, markers } = buildSequence(this.encode, this.ids, state, q, this.config.max_len, this.config.head_max_len);
      if (markers.length !== renderOptions(q).length) {
        throw new Error(`question ${JSON.stringify(qid)}: options do not fit in head_max_len=${this.config.head_max_len} tokens`);
      }
      return { q, ids, markers, qtype: QTYPES[q.t] };
    });

    // rl_common.collate_items: right-pad to the longest sequence / widest option set in the batch
    const n = items.length;
    const L = Math.max(...items.map((it) => it.ids.length));
    const K = Math.max(...items.map((it) => it.markers.length));
    const inputIds = new BigInt64Array(n * L).fill(BigInt(this.ids.pad));
    const attention = new BigInt64Array(n * L);
    const markerPos = new BigInt64Array(n * K);
    const markerMask = new Uint8Array(n * K);
    const qtype = new BigInt64Array(n);
    let nTokens = 0;
    items.forEach((it, i) => {
      it.ids.forEach((v, j) => {
        inputIds[i * L + j] = BigInt(v);
        attention[i * L + j] = 1n;
      });
      nTokens += it.ids.length;
      it.markers.forEach((m, j) => {
        markerPos[i * K + j] = BigInt(m);
        markerMask[i * K + j] = 1;
      });
      qtype[i] = BigInt(it.qtype);
    });

    const out = await this.session.run({
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

    const answers: Record<string, Answer> = {};
    items.forEach((it, r) => {
      const qid = qids[r] as string;
      const k = it.markers.length;
      const temp = this.config.temperature_by_options[tempBucket(it.qtype, k)] ?? this.config.temperature[it.qtype] ?? 1;
      const p = softmax(Array.from(logits.subarray(r * K, r * K + k), (v) => v / temp));
      const ext = { act_probability: actData[r * nAct] ?? 0 };
      const q = it.q;
      if (q.t === "choice") {
        const keys = Object.keys(q.crit as Record<string, string | null>);
        const best = p.indexOf(Math.max(...p));
        answers[qid] = {
          type: "choice",
          choice: keys[best] as string,
          probabilities: Object.fromEntries(keys.map((kk, i) => [kk, round4(p[i] ?? 0)])),
          confidence: round4(confidenceFromProbs(p)),
          rl_agent: ext,
        };
      } else if (q.t === "score") {
        const crit = q.crit as readonly string[];
        answers[qid] = {
          type: "score",
          score: round4(p.reduce((s, v, i) => s + i * v, 0)),
          legend: Object.fromEntries(crit.map((c, i) => [String(i), c])),
          probabilities: Object.fromEntries(p.map((v, i) => [String(i), round4(v)])),
          confidence: round4(confidenceFromProbs(p)),
          rl_agent: ext,
        };
      } else {
        answers[qid] = { type: "noul", noul: round4(p[1] ?? 0), rl_agent: ext };
      }
    });
    return {
      model: "ped",
      answers: answers as SystemOneResult<Q>["answers"],
      usage: { input_tokens: nTokens, output_tokens: 0 },
    };
  }

  /** Release the ONNX session. The instance must not be used afterwards. */
  async close(): Promise<void> {
    await this.session.release();
  }
}
