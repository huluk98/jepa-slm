from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from jepa_slm.config import JepaSettings, ModelShape, TrainingConfig
from jepa_slm.modeling import JepaEncoderDecoder
from jepa_slm.trainer import load_checkpoint, save_checkpoint


def test_checkpoint_loads_with_torch_safe_default(tmp_path: Path) -> None:
    config = TrainingConfig(
        model=ModelShape(
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            d_ff=64,
            attention_heads=4,
            vocab_size=320,
            predictor_width=32,
            predictor_layers=1,
        ),
        jepa=JepaSettings(predictor_width=32, predictor_layers=1),
    )
    model = JepaEncoderDecoder(config.model, config.jepa)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    save_checkpoint(tmp_path, 1, model, optimizer, config)
    checkpoint = torch.load(tmp_path / "step-00000001" / "trainer_state.pt", map_location="cpu")

    assert checkpoint["step"] == 1
    assert checkpoint["config"]["model"]["d_model"] == 32


def test_checkpoint_resume_restores_step_and_weights(tmp_path: Path) -> None:
    config = TrainingConfig(
        model=ModelShape(
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            d_ff=64,
            attention_heads=4,
            vocab_size=320,
            predictor_width=32,
            predictor_layers=1,
        ),
        jepa=JepaSettings(predictor_width=32, predictor_layers=1),
    )
    model = JepaEncoderDecoder(config.model, config.jepa)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with torch.no_grad():
        first_param = next(model.parameters())
        first_param.fill_(0.25)
    save_checkpoint(tmp_path, 7, model, optimizer, config)

    resumed = JepaEncoderDecoder(config.model, config.jepa)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    step = load_checkpoint(
        tmp_path / "step-00000007",
        resumed,
        resumed_optimizer,
        torch.device("cpu"),
    )

    resumed_param = next(resumed.parameters())
    assert step == 7
    assert torch.allclose(resumed_param, torch.full_like(resumed_param, 0.25))
