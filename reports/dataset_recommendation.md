# Dataset Recommendation For JEPA-SLM

Generated: 2026-06-27 12:11

## Recommendation

Use **HuggingFaceTB/smollm-corpus** as the main dataset mixture for the 0.2B JEPA-augmented encoder-decoder model.

For the very first ablation, use **HuggingFaceFW/fineweb-edu `sample-10BT`** because it is small enough to iterate quickly and comes from the same high-quality educational web family that can scale later.

## Ranking

| dataset | score | best use | license |
| --- | --- | --- | --- |
| HuggingFaceTB/smollm-corpus | 94.6 | Main pretraining mixture for the 0.2B model. | ODC-By 1.0 |
| HuggingFaceFW/fineweb-edu sample-10BT | 91.7 | First CE-only vs CE+JEPA pilot. | ODC-By 1.0 plus Common Crawl terms |
| mlfoundations/dclm-baseline-1.0 | 80.1 | General LM baseline if you want DCLM comparability. | CC-BY-4.0 |
| allenai/dolma | 75.7 | Transparency-first research runs. | ODC-By 1.0 plus original source terms |

## Practical Recipe

1. Pilot on `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`.
2. Run `CE-only`, `CE+JEPA-0.10`, and `CE+JEPA-0.25`.
3. If JEPA improves encoder probes without hurting validation CE, move to `HuggingFaceTB/smollm-corpus`.
4. Use a starting mixture of `55% fineweb-edu-dedup`, `20% cosmopedia-v2`, `10% math`, `10% code`, and `5% instruction distillation`.
5. Scale to FineWeb-Edu `sample-100BT` or larger only after the objective ablation is positive.

## Why This Dataset Fits JEPA

JEPA needs target representations that are worth predicting. Educational long-form text, textbooks, tutorials, and code explanations provide stronger semantic structure than random web snippets. That matters because the auxiliary loss asks the student encoder to predict clean contextual hidden states from masked input.

## Candidate Notes

### HuggingFaceTB/smollm-corpus
- URL: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- Scale: Cosmopedia v2, FineWeb-Edu dedup, Python-Edu
- Best use: Main pretraining mixture for the 0.2B model.
- Built around small-model training rather than only 7B+ models.
- Contains educational long-form text, synthetic textbooks, and code tutorials.
- Good for masked source spans and latent representation prediction.

### HuggingFaceFW/fineweb-edu sample-10BT
- URL: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Scale: 10B-token sample; larger 100B/350B/full paths available
- Best use: First CE-only vs CE+JEPA pilot.
- Educational quality filtering is a strong fit for semantic encoder targets.
- The 10B sample is practical for an initial ablation.
- Easy to scale to sample-100BT or the full dataset later.

### mlfoundations/dclm-baseline-1.0
- URL: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- Scale: 4T token / 3B document research baseline
- Best use: General LM baseline if you want DCLM comparability.
- Very strong open-data baseline.
- Less targeted to small encoder-decoder representation learning.
- Dataset card flags research use and weaker domain fit for code/math.

### allenai/dolma
- URL: https://huggingface.co/datasets/allenai/dolma
- Scale: 3T-token documented open corpus with 10B-token sample
- Best use: Transparency-first research runs.
- Excellent documentation and reproducibility story.
- More heterogeneous than FineWeb-Edu or SmolLM-Corpus.
- Useful if dataset transparency matters more than task-targeted small-model quality.

