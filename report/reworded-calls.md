# Reworded validation and test calls

This adds a reworded copy of the 150 validation and 150 test conversations, for the paraphrased, human-reviewed part of item 1 in "Recommended next steps" of `report/transcript-classifier-comparison.md`. It is paraphrased data built from the benchmark's own sentences. It is not an external test set and does not replace one.

Nothing existing is changed. The original `data/benchmark_1000/split.json` is read and never written.

## What is here

| File | Content |
|---|---|
| `data/benchmark_1000_reworded/variants.json` | 108 same-meaning variants (three for each of 36 evidence-bearing sentence types) and 20 near-miss sentences, with the review record |
| `scripts/build_reworded_split.py` | Rebuilds the four conditions below from `split.json` and `variants.json`. Standard library only, seed 20260919 |
| `data/benchmark_1000_reworded/split_para.json` | Evidence sentences replaced by a reviewed variant (2,339 replacements across 300 conversations) |
| `data/benchmark_1000_reworded/split_orig_near.json` | Original wording plus two near-miss sentences per conversation (600 insertions) |
| `data/benchmark_1000_reworded/split_para_near.json` | Both |
| `data/benchmark_1000_reworded/split_para_curly.json` | As `split_para.json` with typographic apostrophes; a punctuation control |
| `data/benchmark_1000_reworded/rebuild_stats.json` | Counts and the fingerprint of the variants file |

Each output keeps the layout of `split.json` (`metadata`, `rows` with `id`, `split`, `conversation`, `truth`) and adds two fields per row: `variants_used` and `near_misses`.

## How it was built

998 of the 1,000 conversations are assembled from 71 stock sentences, 35 spoken by the caller and 36 by the agent. 39 of them carry the evidence for the 27 attributes; they fall into 36 types once numbers and quoted messages are treated as slots.

- **Tier 1, same meaning.** Three variants for each of the 36 types. A variant keeps the fact and changes the words. For attributes whose criterion requires the caller to name the concept (performance, service quality), the variant still names it.
- **Tier 2, near misses.** 20 sentences that sound close to an attribute and do not satisfy its criterion in `BOUNDARIES` (`scripts/jev_noul_benchmark.py`). A near miss for an attribute is inserted only into a conversation whose label for that attribute is 0.
- **Labels.** Carried over from `split.json` unchanged. No label was assigned by hand or by a model.
- **Order of work.** All variants were written, reviewed and frozen before any model was scored on them. The fingerprint of the frozen file is recorded in every output: sha256 `b757a55ac1bb367ae490b3095604816ad75e43977ff495df1c5f8c50166660fa`.

## Human review

One reviewer heard each sentence with its criterion and answered present or absent before seeing the intended label. 144 judgments: 108 tier-1 variants, 20 near misses, 16 negatives used as foils. The first answer matched the intended label on 141 (97.9 percent). The three differences were resolved to the intended label when the criterion was applied word for word. One near miss was judged arguable, rewritten, and reviewed again.

## Limits

1. Only the 39 evidence sentences are reworded. The other 32 stock sentences are unchanged, and 8 of them always accompany a label, so a model can still read some attributes from unchanged text. Any drop measured on this set is a lower bound.
2. The variants were drafted with a language model and then reviewed by a person. They carry that style and are closer to the original register than real speech is.
3. One reviewer, who is also the contributor.
4. Near misses were reviewed against the attribute they imitate. One of them, N12 ("Nobody has come out to look at it yet"), turns on how "explicitly mentioned" is read in the technician-visit criterion, and can fairly be labelled either way. It is kept and flagged; drop it by removing its entry from `variants.json` and rebuilding.
5. The set is still finite and template-driven: 71 sentences with three variants for 36 of them.

## Rebuild

From the repository root:

```
python3 scripts/build_reworded_split.py
```

The five output files are reproduced byte for byte.
