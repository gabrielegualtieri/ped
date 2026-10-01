# PED benchmarks

Reproducible measurements of accuracy, calibration and speed through the public API. The tables in the
main README come from these scripts: every run writes a JSON file to `bench/results/`, and
`yarn bench:report` turns those files into the README's tables. Rerun them on your hardware to compare.

## Accuracy: `bench/massive.ts`

Dataset: the test split of [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_scenario)
(Amazon, CC BY 4.0), voice-assistant requests translated into 51 languages, each labelled with one of
18 scenarios. The same utterance ids are used in every language, so languages and versions are compared
on the same content. Files are downloaded on first run into `bench/.data/`.

Each utterance is one `Router.systemOne` call with three questions:

| Question     | Type               | Correct answer                             |
| ------------ | ------------------ | ------------------------------------------ |
| `scenario`   | choice, 18 options | the utterance's scenario                   |
| `about`      | yes/no             | yes: "Is this request about its scenario?" |
| `aboutOther` | yes/no             | no: the same question for another scenario |

Reported per language and pooled:

- **scenario accuracy**: the top choice is the labelled scenario.
- **scenario ECE**: expected calibration error of the answer's probability (15 bins). 0 means "of the
  answers given with 80% confidence, 80% are right".
- **yes/no accuracy** (threshold 0.5) and **yes/no AUROC** over both yes/no questions. AUROC is
  threshold-free: the chance that a true statement gets a higher P(true) than a false one.
- **p50 / p95 latency** of one call (three questions), on the machine that ran it.

The router chooses the checkpoint without a language hint, as in production. The sample is seeded
(`--n` utterances per language, the same ids everywhere).

```sh
yarn bench:massive --langs en,it,es,zh --n 100
# local bundles instead of downloads
yarn bench:massive --english ./onnx --multilingual ./onnx/multilingual --out bench/results/massive-0.3.0.json
yarn bench:massive --english ./onnx/int8 --multilingual ./onnx/multilingual/int8 --options '{"precision":"int8"}'
```

To compare with an older version on identical requests, check out its tag next to this one and pass its
entry point; `--options '{"revision": ...}'` pins the Hugging Face bundles of that time.
`7f40f52d6d810762b635d963462c26b68241cec3` holds the 0.2 graphs:

```sh
git worktree add ../ped-0.2.0 v0.2.0 && ln -s "$PWD/node_modules" ../ped-0.2.0/node_modules
yarn bench:massive --module ../ped-0.2.0/src/index.ts --options '{"revision":"7f40f52d6d810762b635d963462c26b68241cec3"}'
```

## Speed: `bench/speed.ts`

Load time, resident memory and median latency of one checkpoint: 1, 3 and 10 questions about a short
ticket, one question about a ~450-token message, 32 tickets in one `systemOneBatch`, and with `--long`
one question about a ~2,000-token document, in one sequence (`maxLen: 4096`) and in `systemOneLong`'s
windows. Run one process per configuration, so the memory figure belongs to that checkpoint alone.

```sh
yarn bench:speed --dir ./onnx > bench/results/speed-0.3.0-english.json
yarn bench:speed --dir ./onnx/multilingual --long > bench/results/speed-0.3.0-multilingual.json
yarn bench:speed --options '{"subfolder":"multilingual","precision":"int8"}' --long
```

## Notes

- Latency depends on the CPU, the number of threads ONNX Runtime may use and the input length. The
  result files record the machine they were measured on.
- MASSIVE utterances are short commands; long documents, other domains and other question styles
  behave differently. Validate on your own data before choosing thresholds.
- Each version is measured with the bundles published for it: 0.2.0 with the `sdpa` graphs, 0.3.0 with
  the `eager` graphs (same weights) and with the int8 bundles.
