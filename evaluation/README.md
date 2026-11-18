# Local Hafs evaluation audio

The fixed selection covers 168 professionally recited passages, 84 learner recordings, 48 controlled edits, four samples of everyday Arabic speech and four generated nonspeech signals. An additive selection supplies one clean sibling for each of the 12 edited verses. This makes 320 clips. Only metadata and digests belong in source control.

## Reproduce

Run these commands from the project root using its installed Python environment and a curl binary.

```sh
.venv/bin/python prepare_evaluation_pack.py
.venv/bin/python prepare_evaluation_pack.py --verify
.venv/bin/python prepare_evaluation_pack.py --offline
```

The first command downloads the selected source files, decodes mono 16 kHz PCM, and writes `.cache/evaluation-pack/manifest.json`. An interrupted acquisition can resume. Downloads are content-checked against the release lock. Rebuilding uses the pinned Python dependencies and the source cache. Verification checks clip IDs, bytes and manifest metadata against the shipped lock. The partial manifest is a progress snapshot. Consume the final manifest for reportable results.

## Cohorts and interpretation

| Source | Unit of separation | What it measures |
| --- | --- | --- |
| MP3Quran | 6 development readers and 8 holdout readers. Each reads the same 12 passages. | The official timing catalog specifies the exact Hafs audio and its ayah boundaries. Canonical-reference WER is a proxy. Word timings, verbatim text and error-free delivery have not been independently certified. |
| Reviewed learner recordings | Every excerpt from one parent recording stays in a single split. Speaker identity across recordings is unknown. | The 42 reviewed Hafs excerpts with annotated errors and 42 without registered errors support targeted listening. The published annotations identify canonical sites. Verbatim WER and missed/vs-correct learner errors require an independently checked spoken transcript. |
| Edited human words | The 4 edits and the clean sibling of each ayah stay in one split. The recordings use the same word-asset voice across groups. | Each clip has an exact constructed word sequence and zero-based indices of the heard and reference positions. Repeat, omission, substitution and transposition have independent test cases. Timing is the concatenated asset interval and includes silence. |
| FLEURS | Four Egyptian Arabic test rows in holdout | Published transcripts of everyday speech test the response to Arabic outside Quran recitation. Speech detection and emitting Arabic text can both be correct. Quran pronunciation warnings require separate evidence. |
| Generated nonspeech | Independent signal definitions | Silence, fan-like tone, deterministic noise and impulses test false text emission. |

The two previously examined learner recording IDs are excluded in their entirety. The fixed selection uses every remaining Hafs recording excerpt with a published error and orders the clean candidates by SHA-256 of `hafs-bench-v1/` plus its source filename, taking the first 42. The professional readers are disjoint between splits. Development and holdout share professional target verses to compare voices. Neither this split nor the source inventories establish that the original model developers never saw this material.

## Sources

- MP3Quran supplies the named readers, exact Hafs folders and official ayah boundaries. Individual source URLs, the original time range, and the reciter's name are retained on each row. [Timing catalog](https://www.mp3quran.net/api/v3/ayat_timing/reads) · [Publisher permissions](https://www.mp3quran.net/ar/contact-us)
- The learner collection carries a publisher-declared MIT license. The selection uses revision `d196cd3132c69cd361d28cf6a8b442ae4e0bf0b4`. Use `source_annotations.canonical_word_errors` as sparse reviewer labels. [Learner dataset](https://huggingface.co/datasets/sobolev210/quran-recitation-errors)
- Quran.com word-by-word MP3s supply the educational, local-only controlled edits. Each clip records its source verse, ordered word asset URLs, published spelling and asset boundaries. Audio is excluded from the repository. [Source](https://quran.com) · [Terms](https://quran.com/terms-and-conditions)
- FLEURS is published under CC BY 4.0. Each selected row links its original revision, public asset path and transcript. Temporary signed download URLs are excluded from the manifest. [FLEURS dataset](https://huggingface.co/datasets/google/fleurs)
- The separate QUD timing annex uses CC BY 4.0 reviewed annotations. Each included recording uses byte-verified audio acquired from its identified MP3Quran source. The frozen timing specification records the decision for each of the sixteen QUD recordings. [QUD dataset](https://huggingface.co/datasets/QUD-Technologies/quran-alignment-benchmark)
- The original Quran Lab benchmark requires authenticated access to its audio. This release uses original, publisher-attributed MP3Quran tracks instead. [Quran Lab benchmark](https://huggingface.co/datasets/Quran-Lab/quran-tajweed-benchmark)


## Reviewed timing and rereadings

The additional seven clips contain 107 word occurrences, including 23 repeated occurrences. They cover Ali Jaber reciting Al-Inshiqaq, Al-Husary reciting Al-Fath with phrase rereadings, and Al-Minshawi with the class repeating his recitation of Luqman. QUD Technologies reviewed the original segment sequences and word boundaries. The times are approximate at roughly 50–100 ms and include some padded word endings. The annex adapts selected annotation ranges to three independently sourced MP3Quran recordings. Every excerpt carries the exact source SHA-256, clip boundaries, word IDs, attribution, and three measured offsets from acoustic comparisons across that excerpt. The latest comparison used 4 kHz mono PCM and 3-second correlation samples. All 21 local offsets agree with the applied per-excerpt median within 100 ms. A single offset across the entire Luqman recording is inappropriate because the source and benchmark differ later in the track. [QUD annotation method](https://github.com/QUD-Technologies/quran-alignment-benchmark/blob/33716c28f4f7aa15d76911f1beb32e7d523ab8bd/docs/CORPUS.md) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Run these commands from the project root.

```sh
.venv/bin/python evaluate_timed_recitations.py
.venv/bin/python evaluate_timed_recitations.py --offline --score reports/recitation-timing.json
.venv/bin/python evaluation/verify_timing_sources.py
```

The first command fetches three pinned source recordings and rebuilds the seven WAVs. The second uses both local recognizers and reports raw text, order, repeat retention, timing coverage, and timing deviation against reviewed word occurrences. Edit cost takes precedence in alignment. Among equally good text alignments, timestamps resolve repeated occurrences when available. The timing report gives both matched repeat counts and repeats whose interval actually overlaps the reviewed interval. For the Luqman opening, the report includes both strict written word error and a separate score that accepts the spoken letter names. The final command independently verifies the source identity using small, temporary remote preview ranges. Audio from those previews stays in memory.

## Independent assessment of saved runs

After the offline benchmark completes for both engines, run the following paired comparison for each split.

```sh
.venv/bin/python evaluation/compare_recitation_runs.py --split dev
.venv/bin/python evaluation/compare_recitation_runs.py --split holdout
```

The scorer reads final token IDs, word text, display details, gap anchors, clip hashes and source annotations. It reports distinct final warning occurrences by type, even when two unrecognized ASR words have no corresponding gold position. Coverage measures how often the affected heard word is recognized and graded. A missing-word gap requires two graded neighbors. Weighted word error, per-reader proxy word error, sparse learner-site hits, false warnings on exact structural controls, unjudged learner warnings, and nonspeech/ordinary-speech cases are available in `reports/recitation-release-qa-{dev,holdout}.json`.

## Location, long verses and chapter boundaries

`python evaluation/compare_locations.py` compares both saved splits and writes `reports/upgrade-location.json`. This report separates exact verse IDs, alternate verses with the same fully vocalized words, alternate verses with only the normalized lexical sequence in common, and distinct text. It records abstentions, wrong references that recovered, and the median and 95th percentile of the *source-audio position* where an exact target first appeared. An analysis timeline cannot measure display latency or within-verse word-index correctness.

The independent passage supplement uses an uninterrupted 90 seconds of Ibrahim Al-Akdar’s recitation of 2:282. The publisher's full ayah interval is 205.646 seconds. A second clip joins the end of his 112:3–4 with the original opening and 113:1–2 from the next track. The join is at 9.109 seconds and the first numbered verse of 113 starts at 17.059 seconds. Both source tracks' pauses and the second track's unnumbered opening are retained. Full media and publisher timing hashes are pinned independently of the main pack. These examples measure the sequence of references during an actual long utterance and an edited chapter join. They have no independent verbatim word transcript.

```sh
.venv/bin/python evaluation/passage_challenges.py
.venv/bin/python evaluation/passage_challenges.py --offline --score reports/upgrade-passage-challenges.json
```

## Compare the two Whisper models locally

The optional full large-v3 snapshot is pinned and verified separately. Running these commands leaves the live application's default model unchanged. The growing-window comparison reuses window-level Muaalem evidence from a completed development benchmark in the same checkout. Comparing the same audio and phonetic evidence distinguishes changes introduced by the ASR choice.

```sh
.venv/bin/python compare_dev_asr.py --download-large
.venv/bin/python compare_dev_asr.py --model turbo --output reports/dev-turbo-single.json
.venv/bin/python compare_dev_asr.py --model large --output reports/dev-large-single.json
.venv/bin/python compare_dev_asr.py --compare reports/dev-turbo-single.json reports/dev-large-single.json --output reports/dev-single-comparison.json
.venv/bin/python compare_dev_asr.py --model turbo --growing --output reports/dev-turbo-growing.json
.venv/bin/python compare_dev_asr.py --model large --growing --output reports/dev-large-growing.json
```
## Reproducing the tail, repeat and consonant checks

Run the commands from the project root using the installed models. The repair pack adds 22 clips. The selected reader IDs, learner recording IDs and word edits are in [the selection](repair-voices-v1.json). The [source records](repair-sources-v1.json) contain the publisher metadata for those files and the two additional confirmation readers. The [asset lock](repair-voices-lock.json) pins the source bytes and finished WAV files. The offline rebuild checks both against SHA-256. An audio file used in both automatic and manual mode counts as one clip.

```sh
.venv/bin/python prepare_evaluation_pack.py
.venv/bin/python evaluation/prepare_repair_voices.py
.venv/bin/python evaluation/prepare_repair_voices.py --offline
.venv/bin/python evaluation/evaluate_repair_voices.py --output reports/local-repair-voices.json
```

The evaluator uses the same acoustic windows in both modes and records first-pass ASR, the confirmed transcript, the final warnings and the source interval for each constructed word. To evaluate an archived build on the same inputs, add `--engine-root /path/to/archived/source` and choose a separate output. A matching pair of written words earns the strict repetition score only when the second occurrence alone carries the warning and both occurrences overlap their own source audio. Professional verse timing provides a proxy for expected words. Sparse learner annotations can identify a site for a warning without supplying a verbatim spoken transcript.

For the standard local launch on port 5001, first prepare the chapter-join and speech/noise fixtures. The next two commands send paced PCM over the same WebSocket as the microphone. The tail check exercises the 7.05-second section where the previous release produced «اشتركوا في القناة». The 13-session check records transcript revisions and word-level feedback. The recorded PID identifies the build under test.

```sh
.venv/bin/python evaluation/passage_challenges.py
.venv/bin/python verify_audio_challenges.py --prepare
.venv/bin/python evaluation/verify_live_tail_slice.py --url http://127.0.0.1:5001 --output reports/local-repair-tail.json
.venv/bin/python evaluation/verify_live_repairs.py --url http://127.0.0.1:5001 --server-pid "$(lsof -t -iTCP:5001 -sTCP:LISTEN)" --output reports/local-repair-streams.json
```

The browser suite opens the actual file picker workflow, clicks the affected word, checks its explanation and measures its replay against the word clock in Chromium and WebKit. Run it with a dedicated, idle service. The installed Playwright Core package must have both browsers available.

```sh
node evaluation/verify_repair_browser.mjs --url http://127.0.0.1:5014 --output reports/local-repair-browser.json
```

The follow-up confirmation pack covers two additional readers, two other one-word verses and three single-consonant edits with intact siblings. It was selected after the initial 22-file comparison to test whether two additional changes generalized. The same evaluator scores its 14 waveforms in automatic and manual mode. Its source IDs and promotion criterion are in [the separate selection](repair-confirmation-v1.json).

```sh
.venv/bin/python evaluation/repair_confirmation.py
.venv/bin/python evaluation/repair_confirmation.py --offline
.venv/bin/python evaluation/repair_confirmation.py --score reports/local-repair-confirmation.json
```
