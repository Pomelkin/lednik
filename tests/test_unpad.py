import pytest
import torch

from lednik.models.modeling_lednik import unpad_inputs


def test_unpad_2d_inputs() -> None:
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0], [5, 6, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]])

    unpadded = unpad_inputs(input_ids, attention_mask)

    assert torch.equal(unpadded.unpadded_inputs, torch.tensor([1, 2, 3, 4, 5, 6]))
    assert torch.equal(
        unpadded.cu_seqlens, torch.tensor([0, 3, 4, 6], dtype=torch.int32)
    )
    assert unpadded.max_seqlen == 3
    assert torch.equal(unpadded.non_pad_indices, torch.tensor([0, 1, 2, 4, 8, 9]))


def test_unpad_3d_inputs_keeps_hidden_dim() -> None:
    embeds = torch.randn(2, 3, 8)
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    unpadded = unpad_inputs(embeds, attention_mask)

    assert unpadded.unpadded_inputs.shape == (3, 8)
    torch.testing.assert_close(unpadded.unpadded_inputs[0], embeds[0, 0])
    torch.testing.assert_close(unpadded.unpadded_inputs[2], embeds[1, 0])


def test_to_model_inputs_contains_varlen_keys() -> None:
    input_ids = torch.tensor([[1, 2], [3, 0]])
    attention_mask = torch.tensor([[1, 1], [1, 0]])

    model_inputs = unpad_inputs(input_ids, attention_mask).to_model_inputs()

    assert set(model_inputs) == {
        "input_ids",
        "cu_seqlens",
        "max_seqlen",
        "non_pad_indices",
    }
    assert model_inputs["max_seqlen"] == 2


def test_non_2d_attention_mask_raises() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        unpad_inputs(torch.ones(2, 3), torch.ones(2, 3, 1))
