import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from jepa_slm.config import JepaSettings, ModelShape
from jepa_slm.masking import span_mask_batch
from jepa_slm.modeling import JepaBatch, JepaEncoderDecoder, assert_encoder_only_jepa_contract


def tiny_shape() -> ModelShape:
    return ModelShape(
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        d_ff=64,
        attention_heads=4,
        vocab_size=128,
        predictor_width=32,
        predictor_layers=1,
    )


def tiny_jepa() -> JepaSettings:
    return JepaSettings(
        predictor_width=32,
        predictor_layers=1,
        top_k_target_layers=2,
        lambda_peak=0.25,
    )


def make_batch() -> JepaBatch:
    input_ids = torch.tensor(
        [
            [5, 6, 7, 8, 9, 1, 0, 0],
            [10, 11, 12, 13, 14, 1, 0, 0],
        ]
    )
    attention_mask = input_ids.ne(0).long()
    labels = input_ids.clone()
    labels[labels == 0] = -100
    decoder_input_ids = torch.tensor(
        [
            [0, 5, 6, 7, 8, 9, 1, 0],
            [0, 10, 11, 12, 13, 14, 1, 0],
        ]
    )
    masked = span_mask_batch(
        input_ids,
        attention_mask,
        mask_token_id=2,
        pad_token_id=0,
        mask_fraction=0.4,
        mean_span_length=1,
    )
    return JepaBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_input_ids=decoder_input_ids,
        masked_input_ids=masked.input_ids,
        masked_attention_mask=masked.attention_mask,
        masked_positions=masked.masked_positions,
        masked_position_mask=masked.masked_position_mask,
    )


def test_forward_uses_encoder_only_ema_contract() -> None:
    model = JepaEncoderDecoder(tiny_shape(), tiny_jepa(), pad_token_id=0, eos_token_id=1)
    assert_encoder_only_jepa_contract(model)

    output = model(make_batch(), jepa_weight=0.25)

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.ce_loss)
    assert torch.isfinite(output.jepa_loss)
    assert output.logits.shape[:2] == (2, 8)


def test_ema_encoder_does_not_receive_gradients() -> None:
    model = JepaEncoderDecoder(tiny_shape(), tiny_jepa(), pad_token_id=0, eos_token_id=1)
    output = model(make_batch(), jepa_weight=0.25)
    output.loss.backward()

    assert all(param.grad is None for param in model.ema_encoder.parameters())
    assert any(param.grad is not None for param in model.model.decoder.parameters())
    assert any(param.grad is not None for param in model.predictor.parameters())
