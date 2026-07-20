import torch

from lednik.models import StaticEmbeddingsConfig
from lednik.models import StaticEmbeddingsModel


def _make_model(**config_kwargs) -> StaticEmbeddingsModel:
    config = StaticEmbeddingsConfig(
        vocab_size=50, hidden_size=16, pad_token_id=0, **config_kwargs
    )
    return StaticEmbeddingsModel(config).eval()


def test_forward_shapes() -> None:
    model = _make_model()
    input_ids = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    output = model(input_ids, attention_mask)

    assert output.token_embeddings.shape == (2, 4, 16)
    assert output.sentence_embeddings.shape == (2, 16)


def test_sentence_embedding_is_invariant_to_padding() -> None:
    model = _make_model()
    short = model(torch.tensor([[3, 4]]), torch.tensor([[1, 1]]))
    padded = model(torch.tensor([[3, 4, 0, 0, 0]]), torch.tensor([[1, 1, 0, 0, 0]]))

    torch.testing.assert_close(short.sentence_embeddings, padded.sentence_embeddings)


def test_output_proj_changes_sentence_dim() -> None:
    model = _make_model(output_hidden_size=8)
    output = model(torch.tensor([[3, 4]]), torch.tensor([[1, 1]]))
    assert output.sentence_embeddings.shape == (1, 8)
