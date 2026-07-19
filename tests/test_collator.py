import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend

from lednik.distill.collator import ContrastiveCollator


VOCAB = {"<pad>": 0, "<unk>": 1, "</s>": 2, "hello": 3, "world": 4}
EMB_DIM = 4


@pytest.fixture
def tokenizer() -> TokenizersBackend:
    backend = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
    return TokenizersBackend(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="</s>",
        model_max_length=16,
    )


def _make_collator(tokenizer: TokenizersBackend, **kwargs) -> ContrastiveCollator:
    return ContrastiveCollator(
        tokenizer=tokenizer,
        query_tok_colname="query_tokens",
        query_text_colname="query_text",
        query_teacher_embedding_colname="query_emb",
        pos_tok_colname="pos_tokens",
        pos_text_colname="pos_text",
        pos_teacher_embedding_colname="pos_emb",
        **kwargs,
    )


def _make_item(query_tokens: list[int], pos_tokens: list[int]) -> dict[str, object]:
    return {
        "query_tokens": query_tokens,
        "query_text": "hello world",
        "query_emb": [0.1] * EMB_DIM,
        "pos_tokens": pos_tokens,
        "pos_text": "world",
        "pos_emb": [0.2] * EMB_DIM,
    }


def test_collates_queries_and_positives(tokenizer: TokenizersBackend) -> None:
    collator = _make_collator(tokenizer)
    batch = [_make_item([3, 4], [4]), _make_item([3], [4, 3])]

    output = collator(batch)

    # 2 queries + 2 positives, padded to a multiple of 8
    assert output["input_ids"].shape == (4, 8)
    assert output["attention_mask"].shape == (4, 8)
    # eos is appended to every sequence: [3, 4] -> [3, 4, 2]
    assert output["input_ids"][0, :3].tolist() == [3, 4, 2]
    assert output["attention_mask"].sum(-1).tolist() == [3, 2, 2, 3]

    assert output["queries_mask"].tolist() == [True, True, False, False]
    assert output["positives_mask"].tolist() == [False, False, True, True]
    assert output["negatives_mask"] is None

    assert output["teacher_sentence_embeddings"].shape == (4, EMB_DIM)
    # without label_colname labels are a placeholder of -1 per batch item
    assert torch.equal(output["labels"], torch.tensor([-1, -1]))


def test_padding_uses_pad_token_id(tokenizer: TokenizersBackend) -> None:
    collator = _make_collator(tokenizer)
    output = collator([_make_item([3, 4], [4])])

    lengths = output["attention_mask"].sum(-1)
    for row, length in zip(output["input_ids"], lengths, strict=True):
        assert (row[length:] == tokenizer.pad_token_id).all()


def test_sequences_are_truncated_to_max_len(tokenizer: TokenizersBackend) -> None:
    collator = _make_collator(tokenizer, max_len=4)
    output = collator([_make_item([3, 4, 3, 4, 3, 4], [4])])

    query_row = output["input_ids"][0]
    assert output["attention_mask"][0].sum() == 4
    # truncated to max_len - 1 tokens + eos
    assert query_row[:4].tolist() == [3, 4, 3, 2]


def test_aug_prob_without_corruptor_raises(tokenizer: TokenizersBackend) -> None:
    with pytest.raises(ValueError, match="corruptor"):
        _make_collator(tokenizer, aug_prob=0.5)


def test_partial_negative_columns_raise(tokenizer: TokenizersBackend) -> None:
    with pytest.raises(ValueError, match="together"):
        _make_collator(tokenizer, neg_tok_colname="neg_tokens")
