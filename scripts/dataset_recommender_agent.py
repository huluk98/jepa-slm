#!/usr/bin/env python3
"""Rank open datasets for a 0.2B JEPA-augmented encoder-decoder SLM."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetCandidate:
    name: str
    url: str
    license: str
    scale: str
    best_use: str
    scores: dict[str, float]
    notes: tuple[str, ...]


WEIGHTS = {
    "small_model_fit": 0.25,
    "quality_signal": 0.25,
    "jepa_fit": 0.18,
    "ease_of_pilot": 0.14,
    "license_clarity": 0.10,
    "scale_path": 0.08,
}


DATASETS = (
    DatasetCandidate(
        name="HuggingFaceTB/smollm-corpus",
        url="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus",
        license="ODC-By 1.0",
        scale="Cosmopedia v2, FineWeb-Edu dedup, Python-Edu",
        best_use="Main pretraining mixture for the 0.2B model.",
        scores={
            "small_model_fit": 5,
            "quality_signal": 4.8,
            "jepa_fit": 5,
            "ease_of_pilot": 4.3,
            "license_clarity": 4,
            "scale_path": 4.7,
        },
        notes=(
            "Built around small-model training rather than only 7B+ models.",
            "Contains educational long-form text, synthetic textbooks, and code tutorials.",
            "Good for masked source spans and latent representation prediction.",
        ),
    ),
    DatasetCandidate(
        name="HuggingFaceFW/fineweb-edu sample-10BT",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        license="ODC-By 1.0 plus Common Crawl terms",
        scale="10B-token sample; larger 100B/350B/full paths available",
        best_use="First CE-only vs CE+JEPA pilot.",
        scores={
            "small_model_fit": 4.3,
            "quality_signal": 5,
            "jepa_fit": 4.5,
            "ease_of_pilot": 5,
            "license_clarity": 4,
            "scale_path": 4.4,
        },
        notes=(
            "Educational quality filtering is a strong fit for semantic encoder targets.",
            "The 10B sample is practical for an initial ablation.",
            "Easy to scale to sample-100BT or the full dataset later.",
        ),
    ),
    DatasetCandidate(
        name="mlfoundations/dclm-baseline-1.0",
        url="https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0",
        license="CC-BY-4.0",
        scale="4T token / 3B document research baseline",
        best_use="General LM baseline if you want DCLM comparability.",
        scores={
            "small_model_fit": 3.5,
            "quality_signal": 5,
            "jepa_fit": 3.5,
            "ease_of_pilot": 2.5,
            "license_clarity": 5,
            "scale_path": 5,
        },
        notes=(
            "Very strong open-data baseline.",
            "Less targeted to small encoder-decoder representation learning.",
            "Dataset card flags research use and weaker domain fit for code/math.",
        ),
    ),
    DatasetCandidate(
        name="allenai/dolma",
        url="https://huggingface.co/datasets/allenai/dolma",
        license="ODC-By 1.0 plus original source terms",
        scale="3T-token documented open corpus with 10B-token sample",
        best_use="Transparency-first research runs.",
        scores={
            "small_model_fit": 3.5,
            "quality_signal": 4,
            "jepa_fit": 3.5,
            "ease_of_pilot": 4,
            "license_clarity": 4,
            "scale_path": 4,
        },
        notes=(
            "Excellent documentation and reproducibility story.",
            "More heterogeneous than FineWeb-Edu or SmolLM-Corpus.",
            "Useful if dataset transparency matters more than task-targeted small-model quality.",
        ),
    ),
)


def weighted_score(dataset: DatasetCandidate) -> float:
    return sum(dataset.scores[key] * WEIGHTS[key] for key in WEIGHTS) / 5.0 * 100.0


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def make_report() -> str:
    ranked = sorted(((d, weighted_score(d)) for d in DATASETS), key=lambda item: item[1], reverse=True)
    rows = [
        (dataset.name, f"{score:.1f}", dataset.best_use, dataset.license)
        for dataset, score in ranked
    ]
    top = ranked[0][0]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""# Dataset Recommendation For JEPA-SLM

Generated: {now}

## Recommendation

Use **{top.name}** as the main dataset mixture for the 0.2B JEPA-augmented encoder-decoder model.

For the very first ablation, use **HuggingFaceFW/fineweb-edu `sample-10BT`** because it is small enough to iterate quickly and comes from the same high-quality educational web family that can scale later.

## Ranking

{markdown_table(("dataset", "score", "best use", "license"), rows)}

## Practical Recipe

1. Pilot on `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`.
2. Run `CE-only`, `CE+JEPA-0.10`, and `CE+JEPA-0.25`.
3. If JEPA improves encoder probes without hurting validation CE, move to `HuggingFaceTB/smollm-corpus`.
4. Use a starting mixture of `55% fineweb-edu-dedup`, `20% cosmopedia-v2`, `10% math`, `10% code`, and `5% instruction distillation`.
5. Scale to FineWeb-Edu `sample-100BT` or larger only after the objective ablation is positive.

## Why This Dataset Fits JEPA

JEPA needs target representations that are worth predicting. Educational long-form text, textbooks, tutorials, and code explanations provide stronger semantic structure than random web snippets. That matters because the auxiliary loss asks the student encoder to predict clean contextual hidden states from masked input.

## Candidate Notes

"""


def append_candidate_notes(report: str) -> str:
    chunks = [report]
    for dataset in DATASETS:
        chunks.append(f"### {dataset.name}\n")
        chunks.append(f"- URL: {dataset.url}\n")
        chunks.append(f"- Scale: {dataset.scale}\n")
        chunks.append(f"- Best use: {dataset.best_use}\n")
        for note in dataset.notes:
            chunks.append(f"- {note}\n")
        chunks.append("\n")
    return "".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset recommendation report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_recommendation.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = append_candidate_notes(make_report())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    ranked = sorted(((d, weighted_score(d)) for d in DATASETS), key=lambda item: item[1], reverse=True)
    print(f"Best dataset: {ranked[0][0].name}")
    print(f"Weighted score: {ranked[0][1]:.1f}")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
