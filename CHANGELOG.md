# Changelog

## 0.3.0

More accurate yes/no answers, faster graphs, an int8 option and long documents read in full.
Measurements are in the README's [Benchmarks](README.md#benchmarks) section and can be reproduced with
`bench/`.

### Changed

- **Yes/no questions are asked as a two-option choice** (`noul: "choice"`, the new default of
  `systemOne` and `systemOneBatch`). The checkpoints' noul head answers "no" to most true statements
  (the multilingual one gives a true statement P(true) 0.04 in the median). On MASSIVE in 16 languages
  the yes/no accuracy goes from 60.0% to 70.2% and the AUROC from 0.776 to 0.799. This is the
  workaround upstream documents for the noul head (upstream issue #156). `noul: "native"` restores the 0.2
  answers.
- **Faster graphs.** The published bundles are re-exported with transformers' `eager` attention instead
  of `sdpa`: the same answers (max logit difference ~1e-5) without the NaN guards `sdpa` puts around
  every attention layer. 14% less time per MASSIVE call, up to 28% on a 450-token message and 43% on a
  2,000-token document. Only `ped.onnx` changes; the weights file is byte-identical, so a cached bundle
  downloads 3 MB, also with 0.2.
- Temperatures are clamped to [0.5, 5] at load, as upstream does. The English checkpoint's `choice:11+`
  value (0.10, a 10x sharpener returning point masses, upstream issue #394) becomes 0.5; every other
  shipped value is unchanged. On English MASSIVE requests the 18-option ECE goes from 0.186 to 0.118.
- A conversation state (a list) keeps its newest turns when it has to be cut.
- The state is tokenized once per call instead of once per question.
- An instructions object is serialized as Python's `json.dumps(ensure_ascii=False)`, as upstream does.
- Question `criteria` accept readonly arrays.

### Added

- **`systemOneLong`** reads a document longer than one sequence in overlapping windows instead of
  truncating it (upstream `predict_long`): per question, a yes/no takes the window with the highest
  P(true), a choice or score the most confident window, and `answer.window` names it. Yes/no questions
  use the noul head there by default, whose P(true) stays near 0 on unrelated windows.
- **int8 bundles** (`precision: "int8"`): 30–40% of the download, half to two thirds of the memory and
  1.7x faster per MASSIVE call (about 2x on short requests), within a point of fp32 (56.1% vs 56.2%
  scenario accuracy, 69.4% vs 70.2% yes/no). The MLP down projections stay in fp32: quantizing them as
  well cost the English checkpoint 23 points. fp32 stays the default. `export/export_onnx.py --int8`
  builds them, `--quantize` from an existing bundle.
- **`systemOneBatch`** answers the same questions about many states, sharing ONNX Runtime runs (5–15%
  faster than one call per state on 4 cores). `Router.systemOneBatch` routes every state on its own.
- **`optionOrders` option**: averages each choice over several option orders, removing the position
  bias of a single order (costs one sequence per order).
- **`fitTemperatures` / `setTemperatures` / `temperatures`**: calibrate probabilities on your own labeled
  examples (upstream `fit_temperatures`).
- **`answer_confidence`** on every answer: the probability of the reported answer, the number to gate
  on; `noul` answers also carry `confidence`.
- **Truncation report** in `usage`: `state_tokens`, `state_tokens_dropped`, `truncated`,
  `truncated_questions`, and `options` when a question's options lost their own token span.
- `bench/`: reproducible accuracy (MASSIVE, 16 languages) and speed benchmarks through the public API,
  with the results of every version in `bench/results/`.
- `export/requirements.txt`: the versions the published bundles were exported with.

## 0.2.0

- Multilingual checkpoint (mmBERT-base, 100+ languages) next to the English one.
- `Router`: English text to the English checkpoint, everything else to the multilingual one, each
  loaded on first use; explicit `model` / `lang`, script and language detection otherwise.
- Tokenizer fixes so the multilingual tokenizer produces the Python reference's ids.
- `maxLen` option (the multilingual checkpoint reads up to 8192 tokens).
