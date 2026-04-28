from dataclasses import dataclass
from random import randint
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import SentencePieceBackend
from transformers import TokenizersBackend


@dataclass
class ContrastiveCollator:  # noqa: D101
    tokenizer: TokenizersBackend | SentencePieceBackend
    query_colname: str
    pos_colname: str
    neg_colname: str
    label_colname: str | None = None
    pad_to_multiple_of: int | None = 8
    max_len: int | None = None

    def __post_init__(self):  # noqa: ANN204, D105
        if self.max_len is None:
            self.max_len = int(self.tokenizer.model_max_length)
        # We need to reserve 2 tokens for CLS and SEP, so we subtract 2 from the max_len
        self.max_len = self.max_len - 2
        return

    @staticmethod
    def _select_sentence(sentences: list[list[int]]) -> list[int]:
        if len(sentences) == 1:
            return sentences[0]
        index = randint(0, len(sentences) - 1)  # noqa: S311
        return sentences[index]

    def _preprocess_token_ids(self, token_ids: list[int]) -> Tensor:
        token_ids = token_ids[: self.max_len]
        token_ids = [
            self.tokenizer.cls_token_id,
            *token_ids,
            self.tokenizer.sep_token_id,
        ]
        return torch.tensor(token_ids, dtype=torch.long)

    def _pad_to_multiple_of(self, tensor: Tensor) -> Tensor:
        if self.pad_to_multiple_of is None:
            return tensor
        pad_len = -tensor.size(-1) % self.pad_to_multiple_of
        if pad_len == 0:
            return tensor
        return F.pad(tensor, (0, pad_len), value=self.tokenizer.pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:  # noqa: D102
        queries_list = []
        positives_list = []
        negatives_list = []
        labels_list = []
        for item in batch:
            query = self._preprocess_token_ids(item[self.query_colname])
            pos = self._preprocess_token_ids(
                self._select_sentence(item[self.pos_colname])
            )
            neg = self._preprocess_token_ids(
                self._select_sentence(item[self.neg_colname])
            )
            queries_list.append(query)
            positives_list.append(pos)
            negatives_list.append(neg)

            label = (
                item.get(self.label_colname, None)
                if self.label_colname is not None
                else None
            )
            if label is not None:
                if isinstance(label, int):
                    label = [label]
                labels_list.append(torch.tensor(label, dtype=torch.long).reshape(-1))

        inputs = pad_sequence(
            queries_list + positives_list + negatives_list,
            padding_value=self.tokenizer.pad_token_id,
            batch_first=True,
        )
        inputs = self._pad_to_multiple_of(inputs)

        mask = torch.zeros(inputs.size(0), dtype=torch.long)
        mask[len(queries_list) : len(queries_list) + len(positives_list)] = 1
        mask[len(queries_list) + len(positives_list) :] = 2
        queries_mask = mask == 0
        positives_mask = mask == 1
        negatives_mask = mask == 2

        if len(labels_list) > 0:
            labels = torch.cat(labels_list)
        else:
            labels = torch.full((len(queries_list),), -1, dtype=torch.long)

        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "queries_mask": queries_mask,
            "positives_mask": positives_mask,
            "negatives_mask": negatives_mask,
            "labels": labels,
        }
