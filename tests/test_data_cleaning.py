from jepa_slm.config import DataSettings
from jepa_slm.data import TextStreamDataset, clean_text


def test_clean_text_normalizes_html_controls_and_spaces() -> None:
    assert clean_text("Ａ&nbsp;B\u0000\n C") == "A B C"


def test_clean_text_respects_length_filters() -> None:
    assert clean_text("short", min_chars=10) is None
    assert clean_text("one two three", max_chars=7) == "one two"


def test_stream_dataset_applies_cleaning_to_synthetic_rows() -> None:
    dataset = TextStreamDataset(
        DataSettings(dataset="synthetic", max_samples=1, normalize_text=True, max_chars=12)
    )

    assert next(iter(dataset)) == {"text": "turn on the"}


def test_stream_dataset_reads_local_jsonl_shards(tmp_path) -> None:
    shard = tmp_path / "clean-00000.jsonl"
    shard.write_text('{"text": " keep  this "}\n{"bad": "missing"}\n', encoding="utf-8")
    dataset = TextStreamDataset(
        DataSettings(dataset=str(tmp_path / "clean-*.jsonl"), normalize_text=True)
    )

    assert list(dataset) == [{"text": "keep this"}]
