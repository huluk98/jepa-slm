#!/usr/bin/env python3
"""Research agent for choosing a small language model + JEPA training stack.

The agent is intentionally offline and reproducible: it carries a compact source
ledger, evaluates candidate training architectures against theory-backed
criteria, and emits a markdown report. It is not a web crawler; update the source
ledger when you want to add new papers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    year: int
    url: str
    evidence_type: str
    core_claim: str
    relevance: str


@dataclass(frozen=True)
class Candidate:
    key: str
    name: str
    base_architecture: str
    training_recipe: str
    intended_use: str
    scores: dict[str, float]
    strengths: tuple[str, ...]
    risks: tuple[str, ...]


WEIGHTS = {
    "generative_contract": 0.24,
    "latent_prediction_fit": 0.22,
    "small_model_efficiency": 0.17,
    "theory_support": 0.19,
    "implementation_simplicity": 0.10,
    "ablation_clarity": 0.08,
}


SOURCES = (
    Source(
        key="t5",
        title="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        year=2019,
        url="https://arxiv.org/abs/1910.10683",
        evidence_type="large empirical architecture study",
        core_claim="A text-to-text encoder-decoder can use one maximum-likelihood training interface across many NLP tasks.",
        relevance="Base architecture for small generative encoder-decoder language models.",
    ),
    Source(
        key="bart",
        title="BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension",
        year=2019,
        url="https://arxiv.org/abs/1910.13461",
        evidence_type="seq2seq denoising pretraining",
        core_claim="Corrupt text, then train an encoder-decoder to reconstruct it; the bidirectional encoder and left-to-right decoder serve both comprehension and generation.",
        relevance="Shows why denoising seq2seq remains the generation contract that JEPA should not replace.",
    ),
    Source(
        key="ul2",
        title="UL2: Unifying Language Learning Paradigms",
        year=2022,
        url="https://arxiv.org/abs/2205.05131",
        evidence_type="objective-mixture architecture study",
        core_claim="Mixture-of-Denoisers combines pretraining paradigms and improves the Pareto frontier versus T5- and GPT-like objectives.",
        relevance="Best template for mixing CE denoising, prefix-LM behavior, and auxiliary latent learning.",
    ),
    Source(
        key="data2vec",
        title="data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language",
        year=2022,
        url="https://arxiv.org/abs/2202.03555",
        evidence_type="cross-modal latent prediction",
        core_claim="Predict contextualized latent representations of the full input from a masked view, including for NLP.",
        relevance="Closest direct precedent for text JEPA-style latent target prediction.",
    ),
    Source(
        key="data2vec2",
        title="Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language",
        year=2022,
        url="https://arxiv.org/abs/2212.07525",
        evidence_type="efficiency improvement",
        core_claim="Skipping masked-token encoding, using a fast decoder, and amortizing teacher computation can make latent target learning much cheaper.",
        relevance="Supports a small predictor and careful compute accounting for 0.2B models.",
    ),
    Source(
        key="ijepa",
        title="Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
        year=2023,
        url="https://arxiv.org/abs/2301.08243",
        evidence_type="canonical JEPA implementation",
        core_claim="Predict target-block representations from context-block representations, with EMA target encoder and a predictor.",
        relevance="Provides the JEPA design pattern: context encoder, target encoder, predictor, latent loss.",
    ),
    Source(
        key="vicreg",
        title="VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning",
        year=2021,
        url="https://arxiv.org/abs/2105.04906",
        evidence_type="collapse prevention theory",
        core_claim="Variance and covariance terms can explicitly prevent constant-representation collapse.",
        relevance="Useful as diagnostics or optional regularizers for JEPA hidden states.",
    ),
    Source(
        key="barlow",
        title="Barlow Twins: Self-Supervised Learning via Redundancy Reduction",
        year=2021,
        url="https://arxiv.org/abs/2103.03230",
        evidence_type="redundancy-reduction objective",
        core_claim="Driving cross-correlation toward identity encourages invariance while reducing redundant embedding dimensions.",
        relevance="Mathematical guardrail for representation quality and collapse checks.",
    ),
    Source(
        key="spactor",
        title="SpacTor-T5: Pre-training T5 Models with Span Corruption and Replaced Token Detection",
        year=2024,
        url="https://arxiv.org/abs/2401.13160",
        evidence_type="small/efficient T5 hybrid objective",
        core_claim="A staged hybrid objective can improve T5 pretraining efficiency, with reported reductions in iterations and FLOPs.",
        relevance="Supports staged auxiliary objectives instead of applying all losses equally forever.",
    ),
    Source(
        key="varjepa",
        title="Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture",
        year=2026,
        url="https://arxiv.org/abs/2603.20111",
        evidence_type="probabilistic theory",
        core_claim="JEPA can be interpreted as a deterministic specialization of a latent-variable variational objective.",
        relevance="Gives a mathematical bridge between predictive latent learning and generative modeling theory.",
    ),
)


CANDIDATES = (
    Candidate(
        key="ul2_jepa",
        name="UL2/T5 Encoder-Decoder + Encoder JEPA Auxiliary",
        base_architecture="T5-style pre-norm encoder-decoder, shared embeddings, relative position bias, GEGLU/SwiGLU FFN.",
        training_recipe=(
            "Mixture of span corruption, prefix-LM style supervised CE, and a data2vec/JEPA "
            "latent prediction loss on masked encoder positions."
        ),
        intended_use="Best default for a 0.1B-0.5B seq2seq SLM that must understand input and generate output.",
        scores={
            "generative_contract": 5,
            "latent_prediction_fit": 5,
            "small_model_efficiency": 4,
            "theory_support": 5,
            "implementation_simplicity": 3.5,
            "ablation_clarity": 5,
        },
        strengths=(
            "Keeps token CE as the generation contract.",
            "Gives the encoder a direct semantic prediction task.",
            "Allows staged or mixed denoising objectives.",
            "JEPA components are train-time only and can be removed at inference.",
        ),
        risks=(
            "Too much latent loss can hurt decoder fluency.",
            "EMA target and predictor add training compute.",
            "Needs collapse monitoring, not just loss curves.",
        ),
    ),
    Candidate(
        key="t5_jepa",
        name="Plain T5 Span-Corruption + Encoder JEPA Auxiliary",
        base_architecture="T5-style encoder-decoder with standard span corruption.",
        training_recipe="Span corruption CE plus masked encoder latent prediction against EMA encoder states.",
        intended_use="Good if you want the simplest practical path.",
        scores={
            "generative_contract": 5,
            "latent_prediction_fit": 4.5,
            "small_model_efficiency": 4,
            "theory_support": 4,
            "implementation_simplicity": 4,
            "ablation_clarity": 4.5,
        },
        strengths=(
            "Easy to implement and compare against a normal T5 baseline.",
            "Strong fit for summarization, translation, QA, and instruction-to-output tasks.",
            "Clear ablations: CE-only versus CE+JEPA.",
        ),
        risks=(
            "Less objective diversity than UL2.",
            "May undertrain causal/prefix behavior unless added explicitly.",
        ),
    ),
    Candidate(
        key="bart_jepa",
        name="BART Denoising + Encoder JEPA Auxiliary",
        base_architecture="BART-style bidirectional encoder with autoregressive decoder.",
        training_recipe="Text infilling/noising reconstruction CE plus encoder latent prediction.",
        intended_use="Good for text reconstruction, summarization, and document denoising.",
        scores={
            "generative_contract": 5,
            "latent_prediction_fit": 4,
            "small_model_efficiency": 3.5,
            "theory_support": 4,
            "implementation_simplicity": 3.5,
            "ablation_clarity": 4,
        },
        strengths=(
            "Denoising aligns naturally with masked-context JEPA.",
            "Strong bidirectional encoder plus generation decoder.",
        ),
        risks=(
            "Less unified than UL2 for mixing multiple language modeling modes.",
            "Noising policy can dominate results, making JEPA contribution harder to isolate.",
        ),
    ),
    Candidate(
        key="decoder_only_jepa",
        name="Decoder-Only SLM + Latent Self-Distillation Head",
        base_architecture="GPT-style causal decoder with an auxiliary latent prediction head.",
        training_recipe="Causal LM CE plus hidden-state prediction on dropped spans or future latent states.",
        intended_use="Use only if chat-style incremental decoding matters more than encoder representation quality.",
        scores={
            "generative_contract": 5,
            "latent_prediction_fit": 2.5,
            "small_model_efficiency": 4,
            "theory_support": 2.5,
            "implementation_simplicity": 3,
            "ablation_clarity": 3,
        },
        strengths=(
            "Matches common inference stacks for chat/completion.",
            "No cross-attention path to maintain.",
        ),
        risks=(
            "Causal attention weakens the bidirectional context that makes JEPA/data2vec targets rich.",
            "Auxiliary latent prediction can become redundant with next-token prediction.",
        ),
    ),
    Candidate(
        key="pure_jepa",
        name="Pure Text JEPA Encoder, Decoder Added Later",
        base_architecture="Bidirectional text encoder trained only by latent prediction.",
        training_recipe="JEPA/data2vec latent objective first, then attach or fine-tune a decoder.",
        intended_use="Representation learning or retrieval, not first-choice generative SLM training.",
        scores={
            "generative_contract": 1.5,
            "latent_prediction_fit": 5,
            "small_model_efficiency": 3,
            "theory_support": 4,
            "implementation_simplicity": 2,
            "ablation_clarity": 3,
        },
        strengths=(
            "Cleanest representation-learning experiment.",
            "Useful if the product is embeddings, retrieval, or classification.",
        ),
        risks=(
            "Does not teach token generation.",
            "Later decoder attachment can waste pretraining or require expensive alignment.",
        ),
    ),
)


def weighted_score(candidate: Candidate) -> float:
    return sum(candidate.scores[name] * WEIGHTS[name] for name in WEIGHTS) / 5.0 * 100.0


def ranked_candidates() -> list[tuple[Candidate, float]]:
    return sorted(((c, weighted_score(c)) for c in CANDIDATES), key=lambda item: item[1], reverse=True)


def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def source_table() -> str:
    rows = [
        (s.key, s.year, s.evidence_type, f"[{s.title}]({s.url})")
        for s in SOURCES
    ]
    return table(("key", "year", "evidence type", "source"), rows)


def candidate_table(ranked: list[tuple[Candidate, float]]) -> str:
    rows = []
    for candidate, score in ranked:
        rows.append(
            (
                candidate.name,
                f"{score:.1f}",
                candidate.scores["generative_contract"],
                candidate.scores["latent_prediction_fit"],
                candidate.scores["small_model_efficiency"],
                candidate.scores["theory_support"],
            )
        )
    return table(
        (
            "candidate",
            "weighted score",
            "generation",
            "latent fit",
            "efficiency",
            "theory",
        ),
        rows,
    )


def make_report() -> str:
    ranked = ranked_candidates()
    best = ranked[0][0]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""# JEPA-SLM Research Agent Report

Generated: {now}

## Bottom Line

The best small-language-model training architecture to add JEPA to is:

**{best.name}**

Use JEPA as an **encoder-side auxiliary latent prediction objective**, not as a replacement for token-level language modeling. In plain English: keep the normal seq2seq training that teaches the model to speak, and add a JEPA/data2vec-style loss that teaches the encoder to build better internal world-state representations of text.

Recommended 0.2B shape:

- Base: T5/UL2-style encoder-decoder.
- Width: `d_model=704-768`.
- Depth: `10-12` encoder layers and `8-10` decoder layers.
- FFN: GEGLU/SwiGLU with `d_ff` around `2.5x-4x d_model`.
- Attention: relative position bias or RoPE-style relative handling.
- Predictor: 2-3 narrow Transformer/MLP blocks, about `0.5x-0.7x d_model`.
- EMA target: encoder-only copy, no gradients, no optimizer state.
- Inference: discard EMA target and predictor.

## Candidate Ranking

{candidate_table(ranked)}

## Mathematical Objective

Let `x` be the source sequence, `x_tilde = mask(x)` the corrupted source, `y` the target text, `f_theta` the student encoder, `f_bar` the EMA target encoder, `g_phi` the predictor, and `D_theta` the decoder.

The standard encoder-decoder language-modeling term is:

```text
L_CE(theta) = - sum_t log p_theta(y_t | y_<t, f_theta(x_tilde))
```

The JEPA/data2vec term is:

```text
z_i = stopgrad(norm(mean_top_k(f_bar(x))_i))
h = f_theta(x_tilde)
z_hat_i = norm(g_phi(h, pos_i))

L_JEPA(theta, phi) = (1 / |M|) sum_{{i in M}} || z_hat_i - z_i ||_2^2
```

The EMA target update is:

```text
bar_theta <- tau * bar_theta + (1 - tau) * theta_encoder
```

The total objective:

```text
L_total = L_CE + lambda_jepa(t) * L_JEPA + beta * L_var + gamma * L_cov
```

In the default recipe, `L_var` and `L_cov` are first used as monitors. Add them as small penalties only if representation variance starts to collapse.

Variance monitor:

```text
L_var = (1 / d) sum_j max(0, sigma_min - std_batch_time(z_j))
```

Covariance monitor:

```text
L_cov = (1 / d) sum_{{i != j}} Cov(z)_ij^2
```

## Why This Should Improve A Small Model

### 1. Token CE is sparse supervision for the encoder

In a normal encoder-decoder, the encoder learns only through decoder token loss. That signal is useful, but it is filtered through the decoder's immediate need to predict surface tokens. At small scale, this encourages shortcuts.

The JEPA term adds a direct representation objective: from partial text, predict the clean full-context hidden state. That pushes the encoder toward contextual, paraphrase-stable, semantic features.

### 2. Latent prediction is a better target than raw reconstruction for representation quality

Raw reconstruction asks the model to recover every token detail. Latent prediction asks it to recover the hidden state that the teacher encoder builds from the clean input. The target is still grounded in the text, but it is less obsessed with surface form.

This is the data2vec/JEPAs core move: learn by predicting contextualized representations, not just vocabulary items or pixels.

### 3. The predictor creates a useful bottleneck

If the encoder directly had to match target states, it could contort its hidden space around the auxiliary task. A narrow predictor lets the encoder stay useful for the decoder while the predictor handles the latent matching geometry.

### 4. EMA makes the target stable

The target encoder changes slowly. That makes the latent target less noisy than using the current student as its own target. This is the same stability logic behind mean-teacher/self-distillation methods.

### 5. UL2-style objective mixing protects generation

Pure JEPA does not train the decoder to emit text. UL2/T5-style CE preserves generation, while JEPA improves the hidden states feeding the decoder. The best version is a staged mixture:

```text
phase 1: CE span corruption + JEPA, lambda_jepa ramp 0 -> 0.1
phase 2: CE span corruption + prefix/seq2seq tasks + JEPA, lambda_jepa 0.1 -> 0.25
phase 3: reduce JEPA to 0.05-0.15 or disable for final CE polish
```

## Best Training Loop

```python
for batch in train_loader:
    source_ids, labels = batch["source_ids"], batch["labels"]
    masked_source, masked_positions = span_mask(source_ids)

    with torch.no_grad():
        teacher_layers = ema_encoder(source_ids, output_hidden_states=True)
        target = mean_top_k(teacher_layers, k=4)
        target = layer_norm_no_affine(target)
        target = gather(target, masked_positions)

    student_hidden = model.encoder(masked_source)
    predicted = predictor(student_hidden, masked_positions)
    jepa_loss = mse_or_smooth_l1(l2_normalize(predicted), l2_normalize(target))

    logits = model(
        input_ids=masked_source,
        decoder_input_ids=shift_right(labels),
    ).logits
    ce_loss = cross_entropy(logits, labels)

    loss = ce_loss + lambda_jepa * jepa_loss
    loss.backward()
    optimizer.step()
    update_ema(ema_encoder, model.encoder, tau=ema_tau)
```

## Recommended Ablation

Run these before committing to full training:

1. `CE-only`: normal T5/UL2 span corruption.
2. `CE + JEPA-0.10`: latent loss ramped to 0.10.
3. `CE + JEPA-0.25`: latent loss ramped to 0.25.
4. Optional `CE + JEPA + var/cov`: only if diagnostics show collapse risk.

Success gates:

- Validation CE is no worse than 1-2 percent after warmup.
- Encoder representation variance remains healthy.
- Mean cosine similarity between unrelated examples does not approach 1.
- Encoder probes improve: retrieval, NLI, classification, or semantic similarity.
- Downstream seq2seq metrics match or beat CE-only at equal compute.

## Failure Modes

- **Pure JEPA for generation:** bad idea; it does not teach text emission.
- **Too much lambda:** can improve hidden matching while hurting token quality.
- **Oversized predictor:** the predictor becomes a second model and hides weak encoder learning.
- **Full encoder-decoder EMA:** wasteful at 0.2B; use encoder-only EMA.
- **No diagnostics:** falling JEPA loss can coexist with collapsed representations.

## Source Ledger

{source_table()}

## Source Synthesis

- T5 and BART establish that encoder-decoder denoising is the reliable base for small generative seq2seq models.
- UL2 suggests objective mixing is better than betting on a single pretraining mode.
- data2vec is the strongest text-relevant precedent for predicting contextual latent representations from masked inputs.
- I-JEPA provides the architecture pattern: context encoder, EMA target encoder, predictor, and latent loss.
- VICReg and Barlow Twins explain the collapse problem and give variance/covariance diagnostics.
- SpacTor-T5 supports staged auxiliary objectives for efficient small-model pretraining.
- Var-JEPA connects JEPA to a latent-variable/ELBO interpretation, which supports treating JEPA as a principled latent prediction term rather than a bag of heuristics.
"""


def write_json(path: Path, ranked: list[tuple[Candidate, float]]) -> None:
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": WEIGHTS,
        "ranking": [
            {
                "key": candidate.key,
                "name": candidate.name,
                "weighted_score": round(score, 2),
                "scores": candidate.scores,
                "strengths": candidate.strengths,
                "risks": candidate.risks,
            }
            for candidate, score in ranked
        ],
        "sources": [asdict(source) for source in SOURCES],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a theory-backed recommendation for adding JEPA to a small language model."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/jepa_slm_research_report.md"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("reports/jepa_slm_research_ranking.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked = ranked_candidates()
    report = make_report()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    write_json(args.json, ranked)

    best, score = ranked[0]
    print(f"Best architecture: {best.name}")
    print(f"Weighted score: {score:.1f}")
    print(f"Report written to: {args.report}")
    print(f"JSON ranking written to: {args.json}")


if __name__ == "__main__":
    main()
