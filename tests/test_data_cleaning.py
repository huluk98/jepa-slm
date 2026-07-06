import pytest

from jepa_slm.config import DataSettings
from jepa_slm.data import TextStreamDataset, _neutralize_model_max_length, clean_text


def test_clean_text_normalizes_html_controls_and_spaces() -> None:
    assert clean_text("Ａ&nbsp;B\u0000\n C") == "A B C"


def test_clean_text_respects_length_filters() -> None:
    assert clean_text("short", min_chars=10) is None
    assert clean_text("one two three", max_chars=7) == "one two"


def test_clean_text_drops_format_chars_without_splitting_words() -> None:
    # Soft hyphen / zero-width space / BOM sit inside words; they must vanish,
    # not become word-splitting spaces ("docu ment").
    assert clean_text("docu\u00adment") == "document"
    assert clean_text("zero\u200bwidth\ufeffjoin", normalize=False) == "zerowidthjoin"
    # Control characters (category Cc) still become spaces.
    assert clean_text("tab\there") == "tab here"


def test_min_chars_applies_after_max_chars_cut() -> None:
    # A sparse-space doc cut at max_chars can collapse to almost nothing; the
    # degenerate result must be rejected, not admitted to the corpus.
    text = "title " + "x" * 30_000
    assert clean_text(text, min_chars=200, max_chars=20_000) is None


def test_stream_dataset_applies_cleaning_to_synthetic_rows() -> None:
    dataset = TextStreamDataset(
        DataSettings(dataset="synthetic", max_samples=1, normalize_text=True, max_chars=12)
    )

    assert next(iter(dataset)) == {"text": "turn on the"}


def test_neutralize_model_max_length_lifts_checkpoint_cap() -> None:
    class FakeTokenizer:
        model_max_length = 512

    tokenizer = FakeTokenizer()
    _neutralize_model_max_length(tokenizer)
    # Encoding documents longer than the pretrained cap must not trip HF's
    # "Token indices sequence length is longer than ..." warning path.
    assert tokenizer.model_max_length >= 1_000_000

    bare = object()
    _neutralize_model_max_length(bare)  # tokenizers without the attr are fine


def test_missing_local_corpus_glob_raises_actionable_error(tmp_path) -> None:
    dataset = TextStreamDataset(
        DataSettings(dataset=str(tmp_path / "clean-*.jsonl"), normalize_text=True)
    )

    with pytest.raises(FileNotFoundError, match="prepare_clean_corpus"):
        list(dataset)


def test_stream_dataset_reads_local_jsonl_shards(tmp_path) -> None:
    shard = tmp_path / "clean-00000.jsonl"
    shard.write_text('{"text": " keep  this "}\n{"bad": "missing"}\n', encoding="utf-8")
    dataset = TextStreamDataset(
        DataSettings(dataset=str(tmp_path / "clean-*.jsonl"), normalize_text=True)
    )

    assert list(dataset) == [{"text": "keep this"}]
