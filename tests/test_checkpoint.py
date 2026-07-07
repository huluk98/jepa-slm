from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from jepa_slm.config import JepaSettings, ModelShape, TrainingConfig
from jepa_slm.modeling import JepaEncoderDecoder
from jepa_slm.trainer import load_checkpoint, prune_old_checkpoints, save_checkpoint


def test_prune_old_checkpoints_keeps_last_n(tmp_path: Path) -> None:
    for step in (250, 500, 5000, 10000, 100000):
        (tmp_path / f"step-{step:08d}").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()  # must be ignored

    prune_old_checkpoints(tmp_path, keep_last=2)

    remaining = sorted(p.name for p in tmp_path.glob("step-*"))
    assert remaining == ["step-00010000", "step-00100000"]  # newest 2 by step number
    assert (tmp_path / "not-a-checkpoint").exists()


def test_prune_old_checkpoints_keep_all_when_nonpositive(tmp_path: Path) -> None:
    for step in (1, 2, 3):
        (tmp_path / f"step-{step:08d}").mkdir()
    prune_old_checkpoints(tmp_path, keep_last=0)  # 0 = keep everything
    assert len(list(tmp_path.glob("step-*"))) == 3


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


def _tiny_config() -> TrainingConfig:
    return TrainingConfig(
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


def test_checkpoint_write_is_atomic_and_carries_epoch_position(tmp_path: Path) -> None:
    config = _tiny_config()
    model = JepaEncoderDecoder(config.model, config.jepa)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    save_checkpoint(
        tmp_path, 3, model, optimizer, config, samples_consumed=100, samples_into_epoch=40
    )

    checkpoint_dir = tmp_path / "step-00000003"
    assert not list(checkpoint_dir.glob("*.tmp"))  # temp file renamed away
    state = torch.load(checkpoint_dir / "trainer_state.pt", map_location="cpu")
    assert state["samples_consumed"] == 100
    assert state["samples_into_epoch"] == 40

    resumed = JepaEncoderDecoder(config.model, config.jepa)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    meta: dict = {}
    step = load_checkpoint(
        checkpoint_dir, resumed, resumed_opt, torch.device("cpu"), out_meta=meta
    )
    assert step == 3
    assert meta == {"samples_consumed": 100, "samples_into_epoch": 40}


def test_load_checkpoint_strips_stale_orig_mod_prefix(tmp_path: Path) -> None:
    # Checkpoints written through a torch.compile wrapper before _unwrap_model
    # existed carry "_orig_mod."-prefixed keys; loading must still work.
    config = _tiny_config()
    model = JepaEncoderDecoder(config.model, config.jepa)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    save_checkpoint(tmp_path, 1, model, optimizer, config)

    state_path = tmp_path / "step-00000001" / "trainer_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state["model"] = {f"_orig_mod.{k}": v for k, v in state["model"].items()}
    torch.save(state, state_path)

    resumed = JepaEncoderDecoder(config.model, config.jepa)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    step = load_checkpoint(state_path.parent, resumed, resumed_opt, torch.device("cpu"))
    assert step == 1


def test_save_checkpoint_unwraps_compile_wrapper(tmp_path: Path) -> None:
    config = _tiny_config()
    model = JepaEncoderDecoder(config.model, config.jepa)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    class FakeOptimizedModule:  # mimics torch.compile's wrapper
        def __init__(self, mod):
            self._orig_mod = mod

    save_checkpoint(tmp_path, 2, FakeOptimizedModule(model), optimizer, config)
    state = torch.load(
        tmp_path / "step-00000002" / "trainer_state.pt", map_location="cpu"
    )
    assert not any(k.startswith("_orig_mod.") for k in state["model"])


def test_restore_rng_state_coerces_dtype_and_device() -> None:
    from jepa_slm.trainer import _restore_rng_state

    # A checkpoint loaded with map_location=cuda hands set_rng_state a
    # non-CPU/non-uint8 tensor, which torch rejects; the restore path must
    # coerce it back. Simulate the dtype half on CPU-only machines.
    good_state = torch.get_rng_state()
    _restore_rng_state({"torch": good_state.to(torch.float32)})
    assert torch.equal(torch.get_rng_state(), good_state)
