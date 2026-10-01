/**
 * Tokenizer setup that keeps @huggingface/tokenizers (a JS port) producing the same ids as the Rust
 * `tokenizers` library the Python reference runs on: special-token names read from the checkpoint's
 * tokenizer_config.json, and two fixes for places where the JS port diverges from Rust on the
 * multilingual (Metaspace) tokenizer. The English (ByteLevel) tokenizer is not affected by either.
 */
import type { Tokenizer } from "@huggingface/tokenizers";
import type { SpecialIds } from "./sequence.js";

/** The special-token fields of tokenizer_config.json; a token is a string or an AddedToken object. */
type SpecialToken = string | { content: string };
export type TokenizerConfig = Partial<Record<"cls_token" | "sep_token" | "mask_token" | "pad_token", SpecialToken>>;

/** The part of tokenizer.json this module reads. */
export interface TokenizerJson {
  pre_tokenizer?: { type?: string; replacement?: string; split?: boolean } | null;
}

/**
 * Resolve [CLS]/[SEP]/[MASK]/[PAD] the way the Python reference does (tok.cls_token_id, ...): from the
 * names in tokenizer_config.json, so a checkpoint whose tokenizer uses other names (the multilingual
 * one has <bos>, <eos>, <mask>, <pad>) works too. BERT names are the fallback for configs without them.
 */
export function specialIds(tok: Pick<Tokenizer, "token_to_id">, cfg: TokenizerConfig): SpecialIds {
  const name = (key: keyof TokenizerConfig, fallback: string): string => {
    const v = cfg[key];
    return typeof v === "string" ? v : (v?.content ?? fallback);
  };
  const id = (t: string) => {
    const v = tok.token_to_id(t);
    if (v === undefined) throw new Error(`special token ${t} missing from tokenizer`);
    return v;
  };
  const maskTok = name("mask_token", "[MASK]");
  return { cls: id(name("cls_token", "[CLS]")), sep: id(name("sep_token", "[SEP]")), mask: id(maskTok), pad: id(name("pad_token", "[PAD]")), maskTok };
}

/** The pre-tokenizer and splitter objects inside a @huggingface/tokenizers Tokenizer (not part of its typed API). */
interface PreTokenizer {
  pre_tokenize_text(text: string, options?: object): string[];
}
interface Splitter {
  split(text: string): string[];
}
// the JS port's pre-tokenizers are callable objects: functions carrying the methods
const hasMethod = (x: unknown, name: string): boolean =>
  (typeof x === "object" || typeof x === "function") && x !== null && typeof Reflect.get(x, name) === "function";
const isPreTokenizer = (x: unknown): x is PreTokenizer => hasMethod(x, "pre_tokenize_text");
const isSplitter = (x: unknown): x is Splitter => hasMethod(x, "split");

/**
 * Make a Metaspace tokenizer with `split: true` (the Rust default; the multilingual checkpoint sets
 * it) tokenize as Rust does. Two divergences in the JS port, both around runs of spaces:
 *
 * 1. The JS Metaspace pre-tokenizer ignores `split`. Rust cuts the text before every "▁", so
 *    "▁▁▁leading" is "▁" "▁" "▁leading"; the JS port keeps it whole and BPE merges it to "▁▁▁" "leading".
 * 2. A section between added tokens that normalizes to an added token's text is emitted as that token.
 *    "▁▁" is an added token matched on raw text only, so in Rust the two spaces of "x\n  \ny" stay two
 *    "▁"; the JS port turns them into the single "▁▁" token. Handing it single spaces avoids the lookup.
 *
 * Both are applied to this instance only. Ids then match the Python reference on every string the
 * parity check covers (double spaces, indented e-mail lines, blank lines with spaces, tabs, CJK).
 */
export function alignWithRustTokenizer(tok: Tokenizer, json: TokenizerJson): void {
  const cfg = json.pre_tokenizer;
  if (cfg?.type !== "Metaspace" || cfg.split === false) return;
  const rep = cfg.replacement ?? "\u2581";

  const pt: unknown = Reflect.get(tok, "pre_tokenizer");
  if (isPreTokenizer(pt)) {
    const preTokenize = pt.pre_tokenize_text.bind(pt);
    pt.pre_tokenize_text = (text, options) => preTokenize(text, options).flatMap((s) => splitBefore(s, rep));
  }

  const splitter: unknown = Reflect.get(tok, "splitter_unnormalized");
  const added: unknown = Reflect.get(tok, "added_tokens_map");
  if (isSplitter(splitter) && added instanceof Map) {
    // an added token that strips the whitespace next to it (<mask> has lstrip) takes a whole run of spaces
    const strips = (s: string | undefined, side: "lstrip" | "rstrip"): boolean => {
      const t: unknown = s === undefined ? undefined : added.get(s);
      return typeof t === "object" && t !== null && Reflect.get(t, side) === true;
    };
    const split = splitter.split.bind(splitter);
    splitter.split = (text) => {
      const sections = split(text);
      return sections.flatMap((s, i) => {
        if (s.length < 2 || !/^ +$/.test(s) || strips(sections[i + 1], "lstrip") || strips(sections[i - 1], "rstrip")) return [s];
        return Array.from(s);
      });
    };
  }
}

/** "▁a▁▁b" -> ["▁a", "▁", "▁b"]: each replacement char starts a new piece (Rust's MergedWithNext). */
export function splitBefore(s: string, rep: string): string[] {
  const out: string[] = [];
  let start = 0;
  for (let i = s.indexOf(rep, 1); i !== -1; i = s.indexOf(rep, i + rep.length)) {
    out.push(s.slice(start, i));
    start = i;
  }
  out.push(s.slice(start));
  return out.filter((x) => x.length > 0);
}
