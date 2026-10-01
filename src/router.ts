/**
 * Router: one entry point for every language. It answers with the English checkpoint when the state is
 * English and with the multilingual one (mmBERT, 100+ languages) otherwise, loading each checkpoint on
 * first use. The decision follows laya's `Router._route` (Apache License 2.0, see licenses/laya-LICENSE):
 * explicit model, then explicit language, then script and language detection (lang.ts).
 *
 * Why route instead of always using the multilingual checkpoint: on laya's shared benchmark the English
 * checkpoint is best on English (MASSIVE intent 0.783 vs 0.657), the multilingual one on every other
 * language (0.451 vs 0.306 on 13 languages, XNLI 0.731 vs 0.521), and the English checkpoint stays
 * confident while wrong on scripts it cannot read, so its confidence cannot be used to catch that.
 */
import { analyse, type Detection } from "./lang.js";
import { Ped, type PedOptions } from "./ped.js";
import type { Question, SystemOneResult } from "./types.js";

export type ModelName = "english" | "multilingual";

/** Where each checkpoint lives inside the bundle repo: the English one at the root. */
export const MODEL_SUBFOLDERS: Readonly<Record<ModelName, string | undefined>> = { english: undefined, multilingual: "multilingual" };

export interface RouteOptions {
  /** use this checkpoint and skip detection */
  model?: ModelName;
  /** the state's language when the caller knows it: "it", "en-US", "pt_BR.UTF-8"; "C", "und", "mul" are ignored */
  lang?: string;
}

export interface RouteDecision {
  model: ModelName;
  /** why this checkpoint was chosen */
  reason: string;
  /** the detection result, when detection decided */
  detection: Detection | null;
}

export interface RouterOptions extends Omit<PedOptions, "modelDir" | "subfolder"> {
  /**
   * Checkpoint for a state that gives no clue: no letters, or a short Latin text with no non-English
   * letters ("Grazie mille", "Quero cancelar", "Thanks"). Default "multilingual": English read by the
   * multilingual checkpoint loses some accuracy, while another language read by the English checkpoint
   * gets confidently wrong answers. (laya's Router defaults to "english".)
   */
  default?: ModelName;
  /** per-checkpoint options over the shared ones, e.g. `{ multilingual: { modelDir: "./onnx-multilingual" } }` */
  models?: Partial<Record<ModelName, PedOptions>>;
}

export type RoutedResult<Q extends Record<string, Question>> = SystemOneResult<Q> & { routing: RouteDecision };

const MODELS: readonly ModelName[] = ["english", "multilingual"];
const ENGLISH_SUBTAGS = new Set(["en", "eng", "english"]);
// valid $LANG values that name no language: they abstain instead of pinning a checkpoint
const LANGUAGE_AGNOSTIC = new Set(["c", "posix", "und", "zxx", "mul"]);

function checkModel(model: string): ModelName {
  const m = MODELS.find((x) => x === model);
  if (!m) throw new Error(`unknown model ${JSON.stringify(model)}: expected one of ${MODELS.join(", ")}`);
  return m;
}

/** True/false for a language code ("en", "EN", "en-US", "en_US.UTF-8"), null when it identifies nothing. */
export function englishFromCode(code: string | null | undefined): boolean | null {
  const c = (code ?? "").trim().toLowerCase().split(".")[0] ?? "";
  const primary = c.replaceAll("_", "-").split("-")[0] ?? "";
  if (!primary || LANGUAGE_AGNOSTIC.has(primary)) return null;
  return ENGLISH_SUBTAGS.has(primary);
}

const pct = (x: number) => `${Math.round(100 * x)}%`;

function detectedRoute(det: Detection, fallback: ModelName): RouteDecision {
  const to = (model: ModelName, reason: string): RouteDecision => ({ model, reason, detection: det });
  if (det.script === "unknown") return to(fallback, `no letters detected in state; using default (${fallback})`);
  if (det.script !== "latin") {
    return to("multilingual", `non-Latin script (${det.script}, ${pct(det.nonLatinFraction)} of letters); the English checkpoint cannot read it`);
  }
  if (det.mixedSegment !== null) {
    return to("multilingual", `Latin script, mostly English, but a line or field reads as ${det.language}: ${JSON.stringify(det.mixedSegment.slice(0, 60))}`);
  }
  if (!det.isEnglish && det.language) return to("multilingual", `Latin script, language looks like ${det.language}, not English`);
  if (!det.isEnglish) return to("multilingual", `Latin script, language not identified but ${pct(det.diacriticRate)} non-English letters`);
  if (det.languageUndecided) return to(fallback, `Latin script, language not identified and no non-English letters; using default (${fallback})`);
  return to("english", "English Latin text");
}

/** Pick a checkpoint for `state` without loading anything: explicit model > explicit lang > detection > default. */
export function route(state: unknown, opts: RouteOptions & { default?: ModelName } = {}): RouteDecision {
  const fallback = checkModel(opts.default ?? "multilingual");
  if (opts.model !== undefined) return { model: checkModel(opts.model), reason: `explicit model=${opts.model}`, detection: null };
  const english = englishFromCode(opts.lang);
  if (english !== null) return { model: english ? "english" : "multilingual", reason: `explicit lang=${opts.lang}`, detection: null };
  return detectedRoute(analyse(state), fallback);
}

export class Router {
  private readonly fallback: ModelName;
  private readonly models: Partial<Record<ModelName, PedOptions>>;
  private readonly shared: PedOptions;
  private readonly peds = new Map<ModelName, Promise<Ped>>();

  constructor(opts: RouterOptions = {}) {
    const { default: fallback = "multilingual", models = {}, ...shared } = opts;
    this.fallback = checkModel(fallback);
    this.models = models;
    this.shared = shared;
  }

  /** Which checkpoint `systemOne` would use for this state, and why. Loads nothing. */
  route(state: unknown, opts: RouteOptions = {}): RouteDecision {
    return route(state, { ...opts, default: this.fallback });
  }

  /** The checkpoint's Ped, downloaded and loaded on first use; concurrent callers share one load. */
  load(model: ModelName): Promise<Ped> {
    const name = checkModel(model);
    const loaded = this.peds.get(name);
    if (loaded) return loaded;
    const ped = Ped.load({ ...this.shared, subfolder: MODEL_SUBFOLDERS[name], ...this.models[name] });
    this.peds.set(name, ped);
    // a failed load (offline, ...) is forgotten, so the next call tries again
    void ped.catch(() => {
      if (this.peds.get(name) === ped) this.peds.delete(name);
    });
    return ped;
  }

  /** Load checkpoints up front instead of on the first request that needs them. */
  async preload(models: readonly ModelName[] = MODELS): Promise<void> {
    await Promise.all(models.map((m) => this.load(m)));
  }

  /** Route the state, then answer every question with that checkpoint (see `Ped.systemOne`). */
  async systemOne<Q extends Record<string, Question>>(state: unknown, questions: Q, opts: RouteOptions = {}): Promise<RoutedResult<Q>> {
    const routing = this.route(state, opts);
    const ped = await this.load(routing.model);
    return { ...(await ped.systemOne(state, questions)), routing };
  }

  /** Release every loaded checkpoint. The router can be used again afterwards; it reloads on demand. */
  async close(): Promise<void> {
    const loads = Array.from(this.peds.values());
    this.peds.clear();
    const settled = await Promise.allSettled(loads);
    await Promise.all(settled.flatMap((s) => (s.status === "fulfilled" ? [s.value.close()] : [])));
  }
}
