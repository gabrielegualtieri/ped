import { test } from "node:test";
import assert from "node:assert/strict";
import { Tokenizer } from "@huggingface/tokenizers";
import { alignWithRustTokenizer, specialIds, splitBefore } from "../src/tokenizer.js";

// A Metaspace + BPE tokenizer shaped like the multilingual checkpoint's: spaces normalized to "▁", split
// before every "▁", and "▁▁" both a vocabulary entry and an added token matched on raw text only.
const tinyJson = {
  version: "1.0",
  truncation: null,
  padding: null,
  added_tokens: [
    { id: 0, content: "<pad>", single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
    { id: 1, content: "<mask>", single_word: false, lstrip: true, rstrip: false, normalized: false, special: true },
    { id: 2, content: "\n", single_word: false, lstrip: false, rstrip: false, normalized: false, special: false },
    { id: 3, content: "▁▁", single_word: false, lstrip: false, rstrip: false, normalized: false, special: false },
  ],
  normalizer: { type: "Replace", pattern: { String: " " }, content: "▁" },
  pre_tokenizer: { type: "Metaspace", replacement: "▁", prepend_scheme: "always", split: true },
  post_processor: null,
  decoder: null,
  model: {
    type: "BPE",
    dropout: null,
    unk_token: "<unk>",
    continuing_subword_prefix: null,
    end_of_word_suffix: null,
    fuse_unk: false,
    byte_fallback: false,
    ignore_merges: false,
    vocab: { "<pad>": 0, "<mask>": 1, "\n": 2, "▁▁": 3, "<unk>": 4, "▁": 5, a: 6, b: 7, "▁a": 8, "▁b": 9 },
    merges: [
      ["▁", "▁"],
      ["▁", "a"],
      ["▁", "b"],
    ],
  },
};

// ids from the Rust `tokenizers` library (Python `tokenizers` 0.23) on the same tokenizer.json
const rust: [string, number[]][] = [
  ["a b", [8, 9]],
  ["a  b", [8, 5, 9]],
  ["  a", [5, 8]],
  ["a\n  \nb", [8, 2, 5, 5, 2, 9]],
  ["  ", [5, 5]],
  ["a\n  ", [8, 2, 5, 5]],
  ["  <mask> b", [1, 9]],
  ["a   b", [8, 5, 5, 9]],
];

const encode = (tok: Tokenizer, s: string) => tok.encode(s, { add_special_tokens: false }).ids;

test("alignWithRustTokenizer: runs of spaces tokenize as the Rust library does", () => {
  const tok = new Tokenizer(tinyJson, {});
  // the unpatched JS port merges runs of spaces, so the fix is what makes these pass
  assert.notDeepEqual(
    rust.map(([s]) => encode(tok, s)),
    rust.map(([, ids]) => ids),
  );
  alignWithRustTokenizer(tok, tinyJson);
  for (const [s, ids] of rust) assert.deepEqual(encode(tok, s), ids, JSON.stringify(s));
});

test("alignWithRustTokenizer leaves a non-Metaspace or split: false tokenizer alone", () => {
  const tok = new Tokenizer(tinyJson, {});
  const before = rust.map(([s]) => encode(tok, s));
  alignWithRustTokenizer(tok, { pre_tokenizer: { ...tinyJson.pre_tokenizer, split: false } });
  alignWithRustTokenizer(tok, { pre_tokenizer: { type: "ByteLevel" } });
  assert.deepEqual(
    rust.map(([s]) => encode(tok, s)),
    before,
  );
});

test("splitBefore cuts before every replacement character", () => {
  assert.deepEqual(splitBefore("▁a▁▁b", "▁"), ["▁a", "▁", "▁b"]);
  assert.deepEqual(splitBefore("ab▁c", "▁"), ["ab", "▁c"]);
  assert.deepEqual(splitBefore("▁▁▁", "▁"), ["▁", "▁", "▁"]);
  assert.deepEqual(splitBefore("", "▁"), []);
});

test("specialIds reads the token names from tokenizer_config.json, BERT names as the fallback", () => {
  const vocab = new Map(Object.entries({ "<bos>": 2, "<eos>": 1, "<mask>": 4, "<pad>": 0, "[CLS]": 10, "[SEP]": 11, "[MASK]": 12, "[PAD]": 13 }));
  const tok = { token_to_id: (t: string) => vocab.get(t) };
  assert.deepEqual(specialIds(tok, { cls_token: "<bos>", sep_token: "<eos>", mask_token: { content: "<mask>" }, pad_token: "<pad>" }), {
    cls: 2,
    sep: 1,
    mask: 4,
    pad: 0,
    maskTok: "<mask>",
  });
  assert.deepEqual(specialIds(tok, {}), { cls: 10, sep: 11, mask: 12, pad: 13, maskTok: "[MASK]" });
  assert.throws(() => specialIds(tok, { cls_token: "<s>" }), /special token <s> missing/);
});
