# Dataset Recommendation

## Short Answer

For this JEPA-augmented 0.2B encoder-decoder project, use:

1. **Pilot:** `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`.
2. **Main run:** `HuggingFaceTB/smollm-corpus`.
3. **Scale-up:** `HuggingFaceFW/fineweb-edu`, subset `sample-100BT` or larger.

## Why

JEPA-style latent prediction benefits from text with coherent semantic structure. Educational pages, synthetic textbooks, tutorials, and code explanations are better targets than arbitrary web fragments because the masked context has real structure to infer.

`FineWeb-Edu` provides a high-quality educational web base and has ready-made samples. Its dataset card lists `sample-10BT`, `sample-100BT`, and larger subsets, which makes it ideal for objective ablations before expensive training.

`SmolLM-Corpus` is the best main base because it was designed around small language models. Add small math, code, and instruction-distillation slices so the 0.2B model spends its limited capacity on reasoning-heavy text instead of more generic web.

## Suggested Mix

For the first serious run after the pilot:

| subset | share | purpose |
| --- | --- | --- |
| `fineweb-edu-dedup` | 55% | broad educational language and world knowledge |
| `cosmopedia-v2` | 20% | structured textbook-style explanations |
| math corpus | 10% | symbolic reasoning and word problems |
| code corpus | 10% | procedural reasoning and syntax-heavy text |
| instruction distillation sample | 5% | Qwen-style answer formatting and task behavior |

## Alternatives

`mlfoundations/dclm-baseline-1.0` is a strong open general LM dataset, but it is large and less targeted to encoder-decoder JEPA. Use it if you want DCLM comparability.

`allenai/dolma` is excellent for transparent research and reproducibility, but it is more heterogeneous. Use it if documentation and source diversity matter more than educational density.

## Sources

- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- SmolLM-Corpus: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- SmolLM blog: https://huggingface.co/blog/smollm
- DCLM-Baseline: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- Dolma: https://huggingface.co/datasets/allenai/dolma
