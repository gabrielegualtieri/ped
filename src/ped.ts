/**
 * Ped — TypeScript port of the checkpoint's rl_agent_api.py on top of onnxruntime-node.
 *
 * `Ped.load()` resolves the ONNX bundle (a local directory, or the published Hugging Face repo, cached
 * under ~/.cache/ped), builds the tokenizer, and opens the session. `systemOne()` then answers
 * any number of typed questions about one state in a single forward pass, with the Python
 * reference's sequence layout, per-cardinality temperature and rounding.
 */
import { readFile } from "node:fs/promises";
import path from "node:path";
import * as ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { aggregateWindows, clampedTemperatures } from "./answers.js";
import { fitTemperatureMap, calibrationRecords, type CalibrationExample, type CalibrationOptions, type CalibrationResult } from "./calibrate.js";
import { ensureBundle, type DownloadOptions } from "./download.js";
import { runSequences } from "./infer.js";
import { decodeRequest, encodeRequest, type AnswerOptions, type Item, type NoulMode } from "./requests.js";
import { encodeState, type SpecialIds } from "./sequence.js";
import { alignWithRustTokenizer, specialIds, type TokenizerConfig, type TokenizerJson } from "./tokenizer.js";
import type { PedConfig, Question, SystemOneLongResult, SystemOneResult } from "./types.js";

export type Precision = "fp32" | "int8";

/**
 * Yes/no questions are asked as a two-option choice unless told otherwise: on MASSIVE (16 languages) the
 * noul head answers "no" to most true statements (P(true) around 0.2 for them) and gets 0.55-0.61 right,
 * the two-option choice 0.67-0.74 (bench/). `noul: "native"` keeps the checkpoint's own head.
 */
const DEFAULT_NOUL: NoulMode = "choice";
/** `systemOneLong` keeps the noul head: see there */
const LONG_NOUL: NoulMode = "native";
const PRECISIONS: readonly Precision[] = ["fp32", "int8"];
const NOUL_MODES: readonly NoulMode[] = ["choice", "native"];

export interface PedOptions extends DownloadOptions, AnswerOptions {
  /**
   * Directory holding ped.onnx, ped.onnx.data, ped_config.json and tokenizer/ (the output of
   * export/export_onnx.py). When given, nothing is downloaded and the Hugging Face options are ignored.
   */
  modelDir?: string;
  /**
   * "fp32" (default) or "int8": the int8 bundle (the `int8/` subfolder of the checkpoint) is 30-40% of the
   * download, half to two thirds of the memory and about 2x faster on CPU, within about a point of fp32
   * (README, Benchmarks). Selects what is downloaded; with `modelDir`, point it at the bundle you want.
   */
  precision?: Precision;
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
  /** temperatures over the bundle's, e.g. the result of `fitTemperatures`; every value is clamped to [0.5, 5] */
  temperatures?: Partial<Pick<PedConfig, "temperature" | "temperature_by_options">>;
}

export interface BatchOptions extends AnswerOptions {
  /** most sequences per ONNX Runtime run (a state's questions always share a run); default 64 */
  batchSize?: number;
}

export interface LongOptions extends BatchOptions {
  /** state tokens per window; default: what one question leaves for the state (max_len - head_max_len - 8) */
  window?: number;
  /** tokens between window starts; default window / 2 */
  stride?: number;
}

export class Ped {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly tok: Tokenizer,
    readonly config: PedConfig,
    private readonly ids: SpecialIds,
    /** where the bundle was loaded from */
    readonly modelDir: string,
    private readonly defaults: AnswerOptions,
  ) {}

  static async load(opts: PedOptions = {}): Promise<Ped> {
    if (opts.precision !== undefined && !PRECISIONS.includes(opts.precision)) {
      throw new Error(`precision must be one of ${PRECISIONS.join(", ")}, got ${JSON.stringify(opts.precision)}`);
    }
    // the int8 bundle of a checkpoint sits in its int8/ subfolder
    const subfolder = opts.precision === "int8" ? [opts.subfolder, "int8"].filter(Boolean).join("/") : opts.subfolder;
    const modelDir = opts.modelDir ? path.resolve(opts.modelDir) : await ensureBundle({ ...opts, subfolder });
    const read = async (f: string): Promise<unknown> => JSON.parse(await readFile(path.join(modelDir, f), "utf8"));
    const config = (await read("ped_config.json")) as PedConfig;
    if (opts.maxLen !== undefined) {
      if (!Number.isInteger(opts.maxLen) || opts.maxLen < 1) throw new Error(`maxLen must be a positive integer, got ${opts.maxLen}`);
      config.max_len = opts.maxLen;
    }
    Object.assign(
      config,
      clampedTemperatures({
        temperature: opts.temperatures?.temperature ?? config.temperature,
        temperature_by_options: { ...config.temperature_by_options, ...opts.temperatures?.temperature_by_options },
      }),
    );
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
    return new Ped(session, tok, config, ids, modelDir, { noul: opts.noul, optionOrders: opts.optionOrders });
  }

  private readonly encode = (text: string): number[] => this.tok.encode(text, { add_special_tokens: false }).ids;

  private answerOptions(opts: AnswerOptions = {}): Required<AnswerOptions> {
    const optionOrders = opts.optionOrders ?? this.defaults.optionOrders ?? 1;
    if (!Number.isInteger(optionOrders) || optionOrders < 1) throw new Error(`optionOrders must be a positive integer, got ${optionOrders}`);
    const noul = opts.noul ?? this.defaults.noul ?? DEFAULT_NOUL;
    if (!NOUL_MODES.includes(noul)) throw new Error(`noul must be one of ${NOUL_MODES.join(", ")}, got ${JSON.stringify(noul)}`);
    return { noul, optionOrders };
  }

  /** Answer every question about `state` in one forward pass (Jev's `system_one` request/response shape). */
  async systemOne<Q extends Record<string, Question>>(state: unknown, questions: Q, opts?: AnswerOptions): Promise<SystemOneResult<Q>> {
    const [result] = await this.systemOneBatch([state], questions, opts);
    if (!result) throw new Error("systemOne: no result");
    return result;
  }

  /**
   * Answer the same questions about many states. States of similar length share ONNX Runtime runs (5-15%
   * faster than one `systemOne` per state on 4 cores); results come back in the order of `states`.
   */
  async systemOneBatch<Q extends Record<string, Question>>(states: readonly unknown[], questions: Q, opts: BatchOptions = {}): Promise<SystemOneResult<Q>[]> {
    const ao = this.answerOptions(opts);
    const batchSize = opts.batchSize ?? 64;
    const requests = states.map((state) => encodeRequest(this.encode, this.ids, this.config, state, questions, ao));
    // similar lengths together: less padding
    const longest = (items: Item[]) => Math.max(...items.map((it) => it.ids.length));
    const order = requests.map((_, i) => i).sort((a, b) => longest(requests[a]?.items ?? []) - longest(requests[b]?.items ?? []));
    const results: SystemOneResult<Q>[] = new Array<SystemOneResult<Q>>(states.length);
    for (let start = 0; start < order.length;) {
      let end = start;
      let n = 0;
      while (end < order.length && (end === start || n + (requests[order[end] ?? 0]?.items.length ?? 0) <= batchSize)) {
        n += requests[order[end] ?? 0]?.items.length ?? 0;
        end++;
      }
      const chunk = order.slice(start, end);
      const out = await runSequences(
        this.session,
        this.ids.pad,
        chunk.flatMap((i) => requests[i]?.items ?? []),
      );
      let offset = 0;
      for (const i of chunk) {
        const req = requests[i];
        if (!req) continue;
        const rows = req.items.map((_, t) => ({ logits: out.logits[offset + t] ?? [], act: out.act[offset + t] ?? 0 }));
        offset += req.items.length;
        const { answers, usage } = decodeRequest(this.config, req.qids, req.items, rows);
        results[i] = { model: "ped", answers: answers as SystemOneResult<Q>["answers"], usage };
      }
      start = end;
    }
    return results;
  }

  /**
   * Answer questions about a state longer than one sequence by reading it in overlapping windows
   * (laya `predict_long`): no part of the document is dropped. Per question, a yes/no takes the window
   * with the highest P(true) and a choice or score the most confident window; `answer.window` names it.
   *
   * Yes/no questions use the checkpoint's noul head here unless `noul` is set (at load or per call): the
   * highest P(true) over many windows needs a P(true) that stays near 0 on unrelated text. The noul head
   * gives under 0.03 on unrelated windows, the two-option choice up to 0.99 on some of them.
   * A state that fits one window gets exactly `systemOne`'s answers with the same `noul`.
   */
  async systemOneLong<Q extends Record<string, Question>>(state: unknown, questions: Q, options: LongOptions = {}): Promise<SystemOneLongResult<Q>> {
    const opts = { ...options, noul: options.noul ?? this.defaults.noul ?? LONG_NOUL };
    const budget = opts.window && opts.window > 0 ? opts.window : Math.max(64, this.config.max_len - this.config.head_max_len - 8);
    const stateIds = encodeState(this.encode, this.ids, state);
    if (stateIds.length <= budget) {
      const r = await this.systemOne(state, questions, opts);
      return { ...r, usage: { ...r.usage, truncated: Number(r.usage.truncated), windows: 1 } };
    }
    const step = opts.stride && opts.stride > 0 ? opts.stride : Math.max(1, Math.floor(budget / 2));
    const windows: string[] = [];
    const spans: [number, number][] = [];
    for (let i = 0; i < stateIds.length; i += step) {
      // decoded back to text and re-tokenized as a normal state; the overlap absorbs boundary drift. No
      // clean-up: it would glue punctuation to words ("a ." -> "a."), and transformers skips it for BPE too
      windows.push(this.tok.decode(stateIds.slice(i, i + budget), { clean_up_tokenization_spaces: false }));
      spans.push([i, Math.min(i + budget, stateIds.length)]);
      if (i + budget >= stateIds.length) break;
    }
    const results = await this.systemOneBatch(windows, questions, opts);
    const { answers, usage } = aggregateWindows(Object.keys(questions), results, spans, stateIds.length);
    return { model: "ped", answers: answers as SystemOneLongResult<Q>["answers"], usage };
  }

  /**
   * Fit temperatures on labeled examples (NLL per question type and option count, laya's
   * `fit_temperatures`). Returns them; apply with `setTemperatures` or `Ped.load({ temperatures })`.
   */
  async fitTemperatures(examples: readonly CalibrationExample[], opts: CalibrationOptions = {}): Promise<CalibrationResult> {
    const ao = { ...this.answerOptions(opts), optionOrders: 1 };
    const records = [];
    for (const ex of examples) {
      const { qids, items } = encodeRequest(this.encode, this.ids, this.config, ex.state, ex.questions, ao);
      const out = await runSequences(this.session, this.ids.pad, items);
      records.push(...calibrationRecords(qids, items, out.logits, ex.labels));
    }
    return fitTemperatureMap(records, this.config, opts);
  }

  /** Replace the temperatures (each clamped to [0.5, 5]); a per-cardinality value wins over its type's. */
  setTemperatures(t: Partial<Pick<PedConfig, "temperature" | "temperature_by_options">>): void {
    Object.assign(
      this.config,
      clampedTemperatures({
        temperature: t.temperature ?? this.config.temperature,
        temperature_by_options: t.temperature_by_options ?? this.config.temperature_by_options,
      }),
    );
  }

  /** Release the ONNX session. The instance must not be used afterwards. */
  async close(): Promise<void> {
    await this.session.release();
  }
}
