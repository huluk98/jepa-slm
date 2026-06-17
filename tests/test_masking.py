import pytest

torch = pytest.importorskip("torch")

from jepa_slm.masking import gather_positions, span_mask_batch


def test_span_mask_returns_source_positions_only() -> None:
    input_ids = torch.tensor([[5, 6, 7, 8, 0], [9, 10, 11, 0, 0]])
    attention_mask = input_ids.ne(0).long()

    masked = span_mask_batch(
        input_ids,
        attention_mask,
        mask_token_id=2,
        pad_token_id=0,
        mask_fraction=0.5,
        mean_span_length=1,
    )

    assert masked.input_ids.shape == input_ids.shape
    assert masked.masked_positions.shape == masked.masked_position_mask.shape
    assert masked.masked_position_mask.any()
    for row in range(input_ids.size(0)):
        positions = masked.masked_positions[row][masked.masked_position_mask[row]]
        assert torch.all(attention_mask[row, positions] == 1)
        assert torch.all(masked.input_ids[row, positions] == 2)


def test_gather_positions_batchwise() -> None:
    hidden = torch.arange(2 * 4 * 3).view(2, 4, 3)
    positions = torch.tensor([[0, 2], [1, 3]])

    gathered = gather_positions(hidden, positions)

    assert gathered.shape == (2, 2, 3)
    assert torch.equal(gathered[0, 0], hidden[0, 0])
    assert torch.equal(gathered[0, 1], hidden[0, 2])
    assert torch.equal(gathered[1, 0], hidden[1, 1])
    assert torch.equal(gathered[1, 1], hidden[1, 3])
