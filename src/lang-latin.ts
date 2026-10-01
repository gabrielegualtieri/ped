/**
 * The Latin-script language guess of lang.ts (laya/lang.py `latin_profile`, Apache License 2.0, see
 * licenses/laya-LICENSE), and the Python string semantics the port needs: code-point lengths, `str.split`
 * / `str.strip` whitespace, `str.isupper`, and the `\w`-based word patterns.
 */
import { NON_EN_DIACRITICS, STOP } from "./lang-data.js";

/** Python's `[^\W\d_]+`: letters and letter-like numerals (`\w` minus decimal digits and `_`). */
export const WORD = /[\p{L}\p{Nl}\p{No}]+/gu;
// A token whose dot or @ joins word characters is an identifier, not prose (`github.com`, `v1.2.3`):
// split into words it collides with function words (`com` is Portuguese). The lookbehind keeps it linear.
const IDENTIFIER = /(?<![\p{L}\p{N}_-])[\p{L}\p{N}_-]*(?:[.@][\p{L}\p{N}_-]+)+/gu;
// Python's str.isspace(): JS \s lacks \x1c-\x1f and \x85, and adds
const PY_SPACE = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_SPLIT = new RegExp(`[${PY_SPACE}]+`, "u");
const PY_STRIP = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "gu");

/** length in code points, as Python's len(): a surrogate pair is one character */
export const cpLength = (s: string): number => s.length - (s.match(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g)?.length ?? 0);
export const cpSlice = (s: string, n: number): string => (s.length <= n ? s : Array.from(s).slice(0, Math.max(0, n)).join(""));
export const isAlpha = (ch: string): boolean => /\p{L}/u.test(ch);
/** Python's str.isupper(): at least one uppercase character and no lowercase or titlecase one */
export const isPyUpper = (s: string): boolean => /\p{Uppercase}/u.test(s) && !/[\p{Lowercase}\p{Lt}]/u.test(s);
export const pyStrip = (s: string): string => s.replace(PY_STRIP, "");
export const pySplit = (s: string): string[] => pyStrip(s).split(PY_SPLIT).filter(Boolean);

/** Words more than one list claims (`la`, `un`, `e`): matching one says "not English" without saying which. */
const SHARED_WORDS: ReadonlySet<string> = (() => {
  const seen = new Map<string, number>();
  for (const words of STOP.values()) for (const w of words) seen.set(w, (seen.get(w) ?? 0) + 1);
  return new Set(Array.from(seen).flatMap(([w, n]) => (n > 1 ? [w] : [])));
})();
/** English function words no other list holds: only these carry the English rescue below. */
const EN_ONLY_WORDS: ReadonlySet<string> = new Set(Array.from(STOP.get("en") ?? []).filter((w) => !SHARED_WORDS.has(w)));
/** Function words that are also ordinary English words: a repeat counts once ("do more, do less"). */
const EN_COLLISION_WORDS: ReadonlySet<string> = new Set(
  ["come", "son", "do", "care", "todo", "im", "per", "plus"].filter((w) => Array.from(STOP).some(([lg, sw]) => lg !== "en" && sw.has(w))),
);

// One accented loanword (`café`) must not pull plain English off the English checkpoint: two English-only
// function words and at most one word with a non-English letter rescue it, below this diacritic rate.
const ENGLISH_RESCUE_DIACRITIC_RATE = 0.06;
const NON_EN_DIACRITIC_RATE = 0.02;

function englishRescuedByWords(words: string[], diacRate: number): boolean {
  if (diacRate >= ENGLISH_RESCUE_DIACRITIC_RATE) return false;
  const unique = new Set(words);
  if (Array.from(unique).filter((w) => EN_ONLY_WORDS.has(w)).length < 2) return false;
  return Array.from(unique).filter((w) => Array.from(w).some((ch) => NON_EN_DIACRITICS.has(ch))).length <= 1;
}

export interface LatinProfile {
  /** language code, or null when undecided ("undecided" and "English" are different answers) */
  language: string | null;
  englishHits: number;
  diacriticRate: number;
  looksNonEnglish: boolean;
}

/** Function-word score per language: hits count, except that a collision word counts once. */
function languageScores(counts: Map<string, number>): Map<string, number> {
  const scores = new Map<string, number>();
  for (const [lg, sw] of STOP) {
    let s = 0;
    for (const [w, n] of counts) if (sw.has(w)) s += EN_COLLISION_WORDS.has(w) ? 1 : n;
    scores.set(lg, s);
  }
  return scores;
}

/** Best non-English language that matched at least one word of its own, first on ties. */
function bestEvidenced(counts: Map<string, number>, scores: Map<string, number>): [string | null, number] {
  let bestLg: string | null = null;
  let best = 0;
  for (const [lg, sw] of STOP) {
    if (lg === "en" || !Array.from(counts.keys()).some((w) => sw.has(w) && !SHARED_WORDS.has(w))) continue;
    const s = scores.get(lg) ?? 0;
    if (bestLg === null || s > best) [bestLg, best] = [lg, s];
  }
  return [bestLg, best];
}

/** Share of the (lowercased) characters that ordinary English does not use. */
function diacriticRateOf(text: string): number {
  let diac = 0;
  let len = 0;
  for (const ch of text.toLowerCase()) {
    len++;
    if (NON_EN_DIACRITICS.has(ch)) diac++;
  }
  return diac / Math.max(1, len);
}

/** Python's `_WORD.findall(text)` */
export const findWords = (text: string): string[] => text.match(WORD) ?? [];

/**
 * Evidence behind the Latin-script language guess. A non-English language is named only with a margin
 * over English function words and at least one word no other list claims; otherwise English when English
 * function words are there and nothing non-English outweighs them; otherwise undecided.
 */
export function latinProfile(text: string): LatinProfile {
  // 'İ'.toLowerCase() is 'i' + a combining dot, which matches no word list
  const words = findWords(text.replace(IDENTIFIER, " ").replaceAll("İ", "i").toLowerCase());
  const diacriticRate = diacriticRateOf(text);
  const looksNonEnglish = diacriticRate >= NON_EN_DIACRITIC_RATE;
  if (words.length < 4) return { language: null, englishHits: 0, diacriticRate, looksNonEnglish };

  const counts = new Map<string, number>();
  for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
  const scores = languageScores(counts);
  const en = scores.get("en") ?? 0;
  const [bestLg, best] = bestEvidenced(counts, scores);

  let language: string | null = null;
  if (bestLg !== null && best >= Math.max(2, en + 2)) language = bestLg;
  else if (bestLg !== null && looksNonEnglish && best >= Math.max(2, en)) language = bestLg;
  else if (en > 0 && (!looksNonEnglish || englishRescuedByWords(words, diacriticRate))) language = "en";
  return { language, englishHits: en, diacriticRate, looksNonEnglish };
}
