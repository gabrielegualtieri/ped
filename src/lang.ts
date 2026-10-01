/**
 * Dependency-free script and language detection, used by the Router to pick between the English and
 * the multilingual checkpoint. A port of laya/lang.py from the `laya` Python package 0.3.22 by Convai
 * Innovations (Apache License 2.0, see licenses/laya-LICENSE), kept decision-for-decision identical:
 * test/test_lang.ts pins cases, and a differential run against the Python module agreed on every state.
 *
 * Routing needs one bit: is this English Latin text, or text the English checkpoint cannot read? The
 * English checkpoint collapses on non-Latin scripts (near random, at high confidence) and holds up better
 * on Latin-script languages, so script is the main signal and "is this Latin text English" the second.
 * Script detection is exact; the Latin-language guess is a best-effort stopword/diacritic heuristic.
 *
 * Lengths and slices count code points, as Python's do, so the 4000-character windows match.
 */
import { COMBINING, SCRIPT_RANGES, STOP } from "./lang-data.js";
import { cpLength, cpSlice, findWords, isAlpha, isPyUpper, latinProfile, pySplit, pyStrip } from "./lang-latin.js";

export { latinProfile, type LatinProfile } from "./lang-latin.js";

export interface Detection {
  /** dominant script: "latin", "han", "cyrillic", ... or "unknown" when there are no letters */
  script: string;
  /** fraction of letters per script, Latin first */
  scriptProfile: Record<string, number>;
  /** best-effort language code for Latin text ("en", "it", ...), null when undecided or non-Latin */
  language: string | null;
  /** true when the English checkpoint can be expected to read the state */
  isEnglish: boolean;
  languageUndecided: boolean;
  diacriticRate: number;
  nonLatinFraction: number;
  /** the line or field that made a mostly-English state non-English, else null */
  mixedSegment: string | null;
}

// Non-Latin text is not for the English checkpoint even when Latin letters are the plurality: a short
// message needs a large share; a long payload dilutes it, so there a sentence's worth of letters counts.
const NON_LATIN_FRACTION = 0.2;
const NON_LATIN_MIN_FRACTION = 0.1;
const NON_LATIN_MIN_LETTERS = 10;
const NON_EN_DIACRITIC_RATE = 0.02;
const MAX_CHARS = 4000;

const inRanges = (cp: number, ranges: readonly (readonly [number, number])[]) => ranges.some(([lo, hi]) => lo <= cp && cp <= hi);
const isLatinCp = (cp: number, below: number) =>
  cp < below || (0x1e00 <= cp && cp <= 0x1eff) || (0xff21 <= cp && cp <= 0xff3a) || (0xff41 <= cp && cp <= 0xff5a);
const scriptName = (cp: number) => SCRIPT_RANGES.find(([, ranges]) => inRanges(cp, ranges))?.[0];
const countAlpha = (s: string) => Array.from(s).filter(isAlpha).length;
/**
 * Python's round() of a non-negative number: correctly rounded from the exact binary value, ties to
 * even. toFixed is exact too but sends ties up, and x * 10 ** digits picks up error of its own.
 */
const pyRound = (x: number, digits = 0): number => {
  const exact = x.toFixed(digits + 40);
  const cut = exact.length - 40;
  const tie = /^50*$/.test(exact.slice(cut));
  const last = Number(exact[cut - 1] === "." ? exact[cut - 2] : exact[cut - 1]);
  return tie && last % 2 === 0 ? Number(exact.slice(0, cut)) : Number(x.toFixed(digits));
};

/** The string leaves of a state; keys are ignored (they are usually English field names). */
export function iterText(state: unknown, depth = 0): string[] {
  if (depth > 6 || state === null || state === undefined) return [];
  if (typeof state === "string") return [state];
  if (state instanceof Uint8Array) {
    try {
      return [new TextDecoder("utf-8", { fatal: true }).decode(state)];
    } catch {
      return [];
    }
  }
  if (Array.isArray(state)) return state.flatMap((v) => iterText(v, depth + 1));
  if (state instanceof Map) return Array.from(state.values()).flatMap((v) => iterText(v, depth + 1));
  if (typeof state === "object") return Object.values(state).flatMap((v) => iterText(v, depth + 1));
  return [];
}

/** Flatten a state into the text detection reads, at most `maxChars` code points. */
export function stateText(state: unknown, maxChars = MAX_CHARS): string {
  const parts: string[] = [];
  let budget = maxChars;
  for (const leaf of iterText(state)) {
    if (budget <= 0) break;
    const n = cpLength(leaf);
    if (n > budget) {
      parts.push(cpSlice(leaf, budget));
      break;
    }
    parts.push(leaf);
    budget -= n + 1;
  }
  return cpSlice(parts.join(" "), maxChars);
}

/** Letters by script in one pass; Latin last, so a named script wins a tie against it. */
function scriptCounts(text: string): Map<string, number> {
  const counts = new Map<string, number>();
  let latin = 0;
  for (const ch of text) {
    if (!isAlpha(ch)) continue;
    const cp = ch.codePointAt(0) ?? 0;
    if (isLatinCp(cp, 0x02b0)) {
      latin++;
      continue;
    }
    // a letter no range claims counts as "other": an unreadable script must not go to English
    const name = scriptName(cp) ?? "other";
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  counts.set("latin", latin);
  return counts;
}

function scriptFromCounts(counts: Map<string, number>): string {
  let best = "unknown";
  let bestN = 0;
  for (const [name, n] of counts) {
    if (n > bestN) [best, bestN] = [name, n];
  }
  return best;
}

function profileFromCounts(counts: Map<string, number>): Record<string, number> {
  const total = Array.from(counts.values()).reduce((a, b) => a + b, 0);
  const out: Record<string, number> = {};
  if (!total) return out;
  const latin = counts.get("latin") ?? 0;
  if (latin) out["latin"] = latin / total;
  for (const [name, n] of counts) if (name !== "latin" && n) out[name] = n / total;
  return out;
}

/** Dominant script of `text`: "latin", "han", "devanagari", ... or "unknown" if there are no letters. */
export const detectScript = (text: string): string => scriptFromCounts(scriptCounts(text));

/**
 * Non-Latin runs that read as words rather than annotation inside English prose: a symbol (`α`, one
 * letter), a capitalised proper name (`Дмитрий`) and a pronunciation (no script range claims IPA) are
 * not. A combining mark belongs to the letter before it and never splits a word.
 */
export function nonLatinWords(text: string): string[] {
  const runs: string[] = [];
  let cur = "";
  let script: string | undefined;
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if (inRanges(cp, COMBINING)) continue;
    const s = isLatinCp(cp, 0x0250) ? undefined : scriptName(cp);
    if (s !== undefined && s === script) {
      cur += ch;
      continue;
    }
    if (cur) runs.push(cur);
    [cur, script] = s === undefined ? ["", undefined] : [ch, s];
  }
  if (cur) runs.push(cur);
  return runs.filter((w) => cpLength(w) >= 2 && !isPyUpper(Array.from(w)[0] ?? ""));
}

// Code is not prose in any language, but split into words it reads as one (`os.path` is Portuguese).
const CODE_LINE = /[=;{}[\]]|[\p{L}\p{N}_]\(/u;
// a whitespace token joining two letters/digits with . _ / \ is an identifier or a compound (`Nav/Com`)
const JOINED = /[\p{L}\p{N}][._/\\][\p{L}\p{N}]/u;
// an all-caps run inside mixed-case text is an acronym or a code (`MON`, `LA`, `EST`)
const LETTER_RUN = /[\p{L}\p{Nl}\p{No}]{2,}/gu;

/** Language code for one non-code line, or null when it names no foreign language. */
export function namedProseLanguage(segment: string): string | null {
  if (!pyStrip(segment) || CODE_LINE.test(segment)) return null;
  let prose = pySplit(segment)
    .filter((tok) => !JOINED.test(tok))
    .join(" ");
  if (/\p{Lowercase}/u.test(prose)) prose = prose.replace(LETTER_RUN, (m) => (isPyUpper(m) ? " " : m));
  const tokens = findWords(prose);
  if (tokens.length < 4) return null;
  const lang = latinProfile(prose).language;
  if (lang === null || lang === "en") return null;
  const own = STOP.get(lang) ?? new Set<string>();
  if (new Set(tokens.map((w) => w.toLowerCase()).filter((w) => own.has(w))).size < 2) return null;
  return lang;
}

/** First line or field that, read on its own, is named a non-English language. Reads at most `maxChars`. */
function nonEnglishSegment(state: unknown, maxChars = MAX_CHARS): [string, string] | null {
  let seen = 0;
  for (const leaf of iterText(state)) {
    for (const line of leaf.split("\n")) {
      if (seen >= maxChars) return null;
      const seg = cpSlice(line, maxChars - seen);
      seen += cpLength(seg);
      const lang = namedProseLanguage(seg);
      if (lang) return [lang, pyStrip(seg)];
    }
  }
  return null;
}

function analyseText(text: string): Detection {
  const counts = scriptCounts(text);
  const prof = profileFromCounts(counts);
  let script = scriptFromCounts(counts);
  const hasProfile = Object.keys(prof).length > 0;
  const nonLatin = hasProfile ? pyRound(1 - (prof["latin"] ?? 0), 4) : 0;
  const nNonLatin = pyRound(nonLatin * countAlpha(text));
  if (
    script === "latin" &&
    nonLatinWords(text).length > 0 &&
    (nonLatin >= NON_LATIN_FRACTION || (nonLatin >= NON_LATIN_MIN_FRACTION && nNonLatin >= NON_LATIN_MIN_LETTERS))
  ) {
    script = Object.keys(prof)
      .filter((s) => s !== "latin")
      .reduce((a, b) => ((prof[b] ?? 0) > (prof[a] ?? 0) ? b : a), "");
  }
  const base = { script, scriptProfile: prof, language: null, languageUndecided: true, diacriticRate: 0, mixedSegment: null };
  if (script === "unknown") return { ...base, isEnglish: true, nonLatinFraction: 0 };
  if (script !== "latin") return { ...base, isEnglish: false, nonLatinFraction: nonLatin };
  const lp = latinProfile(text);
  // undecided is not English: non-English letters are enough to prefer the multilingual checkpoint
  const undecided = lp.language === null;
  return {
    ...base,
    language: lp.language,
    isEnglish: lp.language === "en" || (undecided && !lp.looksNonEnglish),
    languageUndecided: undecided,
    diacriticRate: pyRound(lp.diacriticRate, 4),
    nonLatinFraction: nonLatin,
  };
}

/** Whether one line of a field is itself not safe for the English checkpoint (same bar as the segment scan). */
function lineNonEnglish(sample: string): Detection | null {
  if (!pyStrip(sample) || CODE_LINE.test(sample)) return null;
  const det = analyseText(sample);
  if (det.isEnglish) return null;
  if (det.language !== null && det.language !== "en") return namedProseLanguage(sample) === null ? null : det;
  if (det.script !== "latin" && det.script !== "unknown") {
    return nonLatinWords(sample).length > 0 && countAlpha(sample) >= NON_LATIN_MIN_LETTERS ? det : null;
  }
  const words = findWords(sample);
  return det.languageUndecided && det.diacriticRate >= NON_EN_DIACRITIC_RATE && words.length >= 4 ? det : null;
}

/** A string value that is itself not safe for the English checkpoint, else null; the line with most letters wins. */
function leafNonEnglish(leaf: string): Detection | null {
  let bestN = -1;
  let best: Detection | null = null;
  for (const line of leaf.split("\n")) {
    // shorter lines cannot reach four words or ten letters
    if (cpLength(line) < 7) continue;
    const sample = cpSlice(line, MAX_CHARS);
    const det = lineNonEnglish(sample);
    const n = det ? countAlpha(sample) : -1;
    if (det && n > bestN) [bestN, best] = [n, det];
  }
  return best;
}

/**
 * Full detection for a state (a string, or any JSON value whose string leaves are read).
 *
 * A state that would go to the English checkpoint is checked line by line and field by field: a
 * Portuguese ticket with an English stack trace reads as English as a whole, yet the customer's part
 * is what the questions are about. English sent to multilingual loses a few points; the reverse loses
 * calibration, so one non-English line or field is enough.
 */
export function analyse(state: unknown): Detection {
  let result = analyseText(stateText(state));
  if (result.script === "latin" && result.isEnglish) {
    const leaves = iterText(state);
    const found = leaves.length > 1 || leaves.some((l) => l.includes("\n")) ? nonEnglishSegment(state) : null;
    if (found) result = { ...result, language: found[0], isEnglish: false, languageUndecided: false, mixedSegment: found[1] };
  }
  const plain = typeof state === "string" || state instanceof Uint8Array || state === null || state === undefined;
  if (plain || !result.isEnglish) return result;
  // a structured state can still hide a message past the segment cap, or in a script no list names
  const best = mostLettersNonEnglishLeaf(state);
  return best ? { ...result, language: best.language, isEnglish: false, languageUndecided: best.languageUndecided } : result;
}

function mostLettersNonEnglishLeaf(state: unknown): Detection | null {
  let bestN = -1;
  let best: Detection | null = null;
  for (const leaf of iterText(state)) {
    const det = leafNonEnglish(leaf);
    const n = det ? countAlpha(cpSlice(leaf, MAX_CHARS)) : -1;
    if (det && n > bestN) [bestN, best] = [n, det];
  }
  return best;
}

/** True when the English checkpoint can be expected to read this state. */
export const isEnglish = (state: unknown): boolean => analyse(state).isEnglish;
