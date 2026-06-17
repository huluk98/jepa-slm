from pathlib import Path

import pytest

pytest.importorskip("yaml")

from jepa_slm.config import load_training_config


def test_load_full_h20_config_resolves_nested_model() -> None:
    config = load_training_config(Path("configs/train_h20_8gpu.yaml"))

    assert config.model.d_model == 768
    assert config.model.decoder_layers == 10
    assert config.jepa.target_encoder == "ema_encoder_only"
    assert config.batching.source_length == 512
    assert config.data.dataset == "HuggingFaceFW/fineweb-edu"
