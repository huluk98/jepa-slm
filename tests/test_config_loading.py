from pathlib import Path

from jepa_slm.config import load_training_config


def test_load_full_h20_config_resolves_nested_model() -> None:
    config = load_training_config(Path("configs/train_h20_8gpu.yaml"))

    assert config.model.d_model == 768
    assert config.model.decoder_layers == 10
    assert config.jepa.target_encoder == "ema_encoder_only"
    assert config.batching.source_length == 512
    assert config.data.dataset == "HuggingFaceFW/fineweb-edu"


def test_load_4gpu_h20_config_preserves_stop_file() -> None:
    config = load_training_config(Path("configs/train_h20_4gpu.yaml"))

    assert config.batching.per_gpu_micro_batch_sequences == 64
    assert config.runtime.stop_file == "outputs/jepa-slm-h20-4gpu/STOP"
    assert config.runtime.save_on_stop is True
