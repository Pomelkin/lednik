from typing import TypedDict
from collections.abc import Callable
from dataclasses import dataclass
from random import randint
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import SentencePieceBackend
from transformers import TokenizersBackend
from typing import cast


def _get_postprocessor(
    tokenizer: TokenizersBackend | SentencePieceBackend, max_len: int
) -> Callable[[list[int]], list[int]]:
    if tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None:
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id

        def postprocessor(tokens: list[int]) -> list[int]:
            return [cls_id, *tokens[: max_len - 2], sep_id]

    elif tokenizer.eos_token_id is not None:
        eos_id = tokenizer.eos_token_id

        def postprocessor(tokens: list[int]) -> list[int]:
            return [*tokens[: max_len - 1], eos_id]
    else:
        raise ValueError(
            "Tokenizer must have either cls and sep tokens or an eos token."
        )
    return postprocessor


class CollatorOutput(TypedDict):  # noqa: D101
    input_ids: Tensor
    attention_mask: Tensor
    queries_mask: Tensor
    positives_mask: Tensor
    negatives_mask: Tensor | None
    labels: Tensor
    teacher_sentence_embeddings: Tensor


@dataclass
class ContrastiveCollator:  # noqa: D101
    tokenizer: TokenizersBackend | SentencePieceBackend

    query_tok_colname: str
    query_teacher_embedding_colname: str

    pos_tok_colname: str
    pos_teacher_embedding_colname: str

    neg_tok_colname: str | None = None
    neg_teacher_embedding_colname: str | None = None

    label_colname: str | None = None
    pad_to_multiple_of: int | None = 8
    max_len: int | None = None
    postprocessor: Callable[[list[int]], list[int]] | None = None

    def __post_init__(self):  # noqa: ANN204, D105
        if self.max_len is None:
            self.max_len = int(self.tokenizer.model_max_length)
        self.postprocessor = _get_postprocessor(self.tokenizer, self.max_len)
        if (self.neg_tok_colname is None) != (
            self.neg_teacher_embedding_colname is None
        ):
            raise ValueError(
                "neg_tok_colname and neg_teacher_embedding_colname must be provided together or omitted."
            )
        return

    @staticmethod
    def _select_sentence(sentences: list[list[int]]) -> list[int]:
        if len(sentences) == 1:
            return sentences[0]
        index = randint(0, len(sentences) - 1)  # noqa: S311
        return sentences[index]

    def _preprocess_token_ids(self, token_ids: list[int]) -> Tensor:
        if self.postprocessor is None:
            raise ValueError(
                "Postprocessor must be defined before preprocessing token ids."
            )
        token_ids = self.postprocessor(token_ids)
        return torch.tensor(token_ids, dtype=torch.long)

    def _pad_to_multiple_of(self, tensor: Tensor) -> Tensor:
        if self.pad_to_multiple_of is None:
            return tensor
        pad_len = -tensor.size(-1) % self.pad_to_multiple_of
        if pad_len == 0:
            return tensor
        return F.pad(tensor, (0, pad_len), value=self.tokenizer.pad_token_id)

    def _get_embedding_and_coresponding_tokens(
        self, item: dict[str, Any], tok_colname: str, teacher_emb_colname: str
    ) -> tuple[Tensor, Tensor]:
        tokens = item[tok_colname]
        embedding = item[teacher_emb_colname]
        if isinstance(tokens[0], list):
            index = randint(0, len(tokens) - 1)  # noqa: S311
            tokens = tokens[index]
            embedding = embedding[index]

        tokens = self._preprocess_token_ids(tokens)
        embedding = torch.tensor(embedding)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        return tokens, embedding

    def __call__(self, batch: list[dict[str, Any]]) -> CollatorOutput:  # noqa: D102
        query_tok_list = []
        pos_tok_list = []
        neg_tok_list = []

        query_embeddings = []
        pos_embeddings = []
        neg_embeddings = []

        use_negatives = (
            self.neg_tok_colname is not None
            and self.neg_teacher_embedding_colname is not None
        )

        labels_list = []
        for item in batch:
            query_tok, query_emb = self._get_embedding_and_coresponding_tokens(
                item,
                self.query_tok_colname,
                self.query_teacher_embedding_colname,
            )
            pos_tok, pos_emb = self._get_embedding_and_coresponding_tokens(
                item,
                self.pos_tok_colname,
                self.pos_teacher_embedding_colname,
            )
            if use_negatives:
                neg_tok, neg_emb = self._get_embedding_and_coresponding_tokens(
                    item,
                    cast(str, self.neg_tok_colname),
                    cast(str, self.neg_teacher_embedding_colname),
                )
                neg_tok_list.append(neg_tok)
                neg_embeddings.append(neg_emb)

            query_tok_list.append(query_tok)
            pos_tok_list.append(pos_tok)
            query_embeddings.append(query_emb)
            pos_embeddings.append(pos_emb)

            label = (
                item.get(self.label_colname, None)
                if self.label_colname is not None
                else None
            )
            if label is not None:
                if isinstance(label, int):
                    label = [label]
                labels_list.append(torch.tensor(label, dtype=torch.long).reshape(-1))

        sequences = query_tok_list + pos_tok_list
        if use_negatives:
            sequences += neg_tok_list
        inputs = pad_sequence(
            sequences,
            padding_value=self.tokenizer.pad_token_id,
            batch_first=True,
        )
        inputs = self._pad_to_multiple_of(inputs)

        mask = torch.zeros(inputs.size(0), dtype=torch.long)
        mask[len(query_tok_list) : len(query_tok_list) + len(pos_tok_list)] = 1
        if use_negatives:
            mask[len(query_tok_list) + len(pos_tok_list) :] = 2

        queries_mask = mask == 0
        positives_mask = mask == 1
        negatives_mask = mask == 2 if use_negatives else None

        if len(labels_list) > 0:
            labels = torch.cat(labels_list)
        else:
            labels = torch.full((len(query_tok_list),), -1, dtype=torch.long)

        attention_mask = (inputs != self.tokenizer.pad_token_id).long()

        teacher_embeddings = query_embeddings + pos_embeddings
        if use_negatives:
            teacher_embeddings += neg_embeddings
        teacher_sentence_embeddings = torch.cat(teacher_embeddings)
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "queries_mask": queries_mask,
            "positives_mask": positives_mask,
            "negatives_mask": negatives_mask,
            "labels": labels,
            "teacher_sentence_embeddings": teacher_sentence_embeddings,
        }
