import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import TypedDict

import torch
from kostyl.utils import setup_logger
from sage.spelling_corruption import SBSCCorruptor
from tokenizers import Tokenizer
from torch import Tensor
from transformers import SentencePieceBackend
from transformers import TokenizersBackend


logger = setup_logger(fmt="only_message")


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


class CollatorOutput(TypedDict):
    """Output structure for the collator."""

    input_ids: Tensor
    attention_mask: Tensor
    queries_mask: Tensor
    positives_mask: Tensor
    negatives_mask: Tensor | None
    labels: Tensor
    teacher_sentence_embeddings: Tensor


@dataclass
class ContrastiveCollator:
    """Collator for contrastive distillation with optional text augmentation."""

    tokenizer: TokenizersBackend

    query_tok_colname: str
    query_text_colname: str
    query_teacher_embedding_colname: str

    pos_tok_colname: str
    pos_text_colname: str
    pos_teacher_embedding_colname: str

    neg_tok_colname: str | None = None
    neg_text_colname: str | None = None
    neg_teacher_embedding_colname: str | None = None

    label_colname: str | None = None
    pad_to_multiple_of: int | None = 8
    max_len: int | None = None

    aug_prob: float = 0.0
    corruptor: SBSCCorruptor | None = None

    backend_tokenizer: Tokenizer | None = None

    def __post_init__(self) -> None:
        """Post-initialization checks and setup for the collator."""
        if self.backend_tokenizer is None:
            if isinstance(self.tokenizer, TokenizersBackend):
                self.backend_tokenizer = self.tokenizer.backend_tokenizer
                self.backend_tokenizer.no_padding()
                self.backend_tokenizer.no_truncation()
            else:
                raise ValueError(
                    "backend_tokenizer must be provided for SentencePieceBackend."
                )
        ### Augmentation checks
        if self.aug_prob > 0.0 and self.corruptor is None:
            raise ValueError(
                "If aug_prob > 0.0, a corruptor must be provided for data augmentation."
            )
        if not 0.0 <= self.aug_prob <= 1.0:
            raise ValueError("aug_prob must be between 0.0 and 1.0.")

        if self.aug_prob == 0.0 and self.corruptor is not None:
            self.corruptor = None
            logger.warning("aug_prob is close to 0.0, setting corruptor to None.")

        ### Max length check
        if self.max_len is None:
            self.max_len = int(self.tokenizer.model_max_length)

        if not (
            (self.neg_tok_colname is None)
            == (self.neg_teacher_embedding_colname is None)
            == (self.neg_text_colname is None)
        ):
            raise ValueError(
                "neg_tok_colname, neg_text_colname, and neg_teacher_embedding_colname must be provided together or omitted."
            )
        return

    def _determine_seqlen(self, max_len_in_batch: int) -> int:
        if self.pad_to_multiple_of is None:
            return max_len_in_batch
        res = -max_len_in_batch % self.pad_to_multiple_of
        return max_len_in_batch + res

    def _pad_sequences(
        self,
        sequences: list[Tensor],
        pad_value: int,
        pad_len: int,
    ) -> Tensor:
        seq_buf = torch.full((len(sequences), pad_len), pad_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            if seq.ndim != 1:
                raise ValueError(f"Expected 1D tensor, got {seq.ndim}D tensor.")
            seq_buf[i, : seq.size(0)] = seq
        return seq_buf

    def _encode_texts(self, texts: list[str]) -> list[list[int]]:
        if self.backend_tokenizer is None:
            raise ValueError("backend_tokenizer must be defined for text encoding.")

        encoded = self.backend_tokenizer.encode_batch_fast(
            texts,
            add_special_tokens=False,
        )
        return [enc.ids for enc in encoded]

    def _postprocess_tokens(self, tokens: list[int]) -> list[int]:
        if self.max_len is None:
            raise ValueError("max_len must be defined for postprocessing tokens.")

        if (
            self.tokenizer.cls_token_id is not None
            and self.tokenizer.sep_token_id is not None
        ):
            cls_id = self.tokenizer.cls_token_id
            sep_id = self.tokenizer.sep_token_id
            tokens = [cls_id, *tokens[: self.max_len - 2], sep_id]
        elif self.tokenizer.eos_token_id is not None:
            eos_id = self.tokenizer.eos_token_id
            tokens = [*tokens[: self.max_len - 1], eos_id]
        else:
            raise ValueError(
                "Tokenizer must have either cls and sep tokens or an eos token."
            )
        return tokens

    def _gather_tokens(  # noqa: C901
        self, items: list[dict[str, Any]], text_colname: str, tokens_colname: str
    ) -> list[Tensor]:
        texts: list[str] = []
        tokens: list[list[int] | None] = []

        if self.aug_prob > 0.0:
            aug_indices = random.sample(
                range(len(items)),
                k=min(math.ceil(len(items) * self.aug_prob), len(items)),
            )
            # Sort indices to maintain order when replacing tokens after encoding
            aug_indices.sort()
        else:
            aug_indices = []

        aug_indices_set = set(aug_indices)
        for i, item in enumerate(items):
            if i in aug_indices_set:
                if self.corruptor is None:
                    raise ValueError(
                        "Corruptor must be provided for data augmentation."
                    )

                item_texts: list[str] = item[text_colname]
                if not isinstance(item_texts, list):
                    raise ValueError(
                        f"Expected a list of texts for augmentation, got {type(item_texts)}."
                    )

                text = (
                    random.choice(item_texts) if len(item_texts) > 1 else item_texts[0]  # noqa: S311
                )
                corrupted_text = self.corruptor.corrupt(text)
                texts.append(corrupted_text)
                tokens.append(None)
            else:
                item_tokens: list[list[int]] = item[tokens_colname]
                if not isinstance(item_tokens, list):
                    raise ValueError(
                        f"Expected a list of tokens, got {type(item_tokens)}."
                    )
                toks: list[int] = (
                    random.choice(item_tokens)  # noqa: S311
                    if len(item_tokens) > 1
                    else item_tokens[0]
                )
                if not isinstance(toks, list) or not isinstance(toks[0], int):
                    raise ValueError(
                        f"Expected a list of integers for tokens, got {toks}."
                    )
                tokens.append(toks)

        if len(texts) > 0:
            encoded_texts = self._encode_texts(texts)
            for idx, enc in zip(aug_indices, encoded_texts, strict=True):
                tokens[idx] = enc

        tokens_t: list[Tensor] = []
        for tok in tokens:
            if tok is None:
                raise ValueError("Tokenization failed for some items.")

            tok = self._postprocess_tokens(tok)
            tokens_t.append(torch.tensor(tok, dtype=torch.long))
        return tokens_t

    def _gather_teacher_embeddings(
        self, items: list[dict[str, Any]], teacher_emb_colname: str
    ) -> Tensor:
        embeddings: list[Tensor] = []
        for item in items:
            emb = torch.tensor(item[teacher_emb_colname])
            if emb.ndim != 1:
                raise ValueError(
                    f"Expected 1D tensor for teacher embedding, got {emb.ndim}D tensor."
                )
            embeddings.append(emb)
        return torch.stack(embeddings)

    def __call__(self, batch: list[dict[str, Any]]) -> CollatorOutput:
        """Collate items into padded inputs, section masks, and teacher embeddings."""
        sequences = self._gather_tokens(
            batch, self.query_text_colname, self.query_tok_colname
        )
        sequences += self._gather_tokens(
            batch, self.pos_text_colname, self.pos_tok_colname
        )
        embeddings = [
            self._gather_teacher_embeddings(
                batch, self.query_teacher_embedding_colname
            ),
            self._gather_teacher_embeddings(batch, self.pos_teacher_embedding_colname),
        ]

        use_negatives = False
        if (
            self.neg_tok_colname is not None
            and self.neg_text_colname is not None
            and self.neg_teacher_embedding_colname is not None
        ):
            use_negatives = True
            sequences += self._gather_tokens(
                batch, self.neg_text_colname, self.neg_tok_colname
            )
            embeddings.append(
                self._gather_teacher_embeddings(
                    batch, self.neg_teacher_embedding_colname
                )
            )

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token for batch collation.")

        pad_len = self._determine_seqlen(max(seq.size(0) for seq in sequences))
        input_ids = self._pad_sequences(
            sequences, pad_value=pad_token_id, pad_len=pad_len
        )

        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
        attention_mask = (torch.arange(pad_len) < lengths.unsqueeze(1)).long()

        batch_size = len(batch)
        total = input_ids.size(0)
        queries_mask = torch.zeros(total, dtype=torch.bool)
        queries_mask[:batch_size] = True
        positives_mask = torch.zeros(total, dtype=torch.bool)
        positives_mask[batch_size : 2 * batch_size] = True
        negatives_mask: Tensor | None = None
        if use_negatives:
            negatives_mask = torch.zeros(total, dtype=torch.bool)
            negatives_mask[2 * batch_size :] = True

        if self.label_colname is not None:
            labels = torch.cat(
                [
                    torch.tensor(item[self.label_colname], dtype=torch.long).reshape(-1)
                    for item in batch
                ]
            )
        else:
            labels = torch.full((batch_size,), -1, dtype=torch.long)

        return CollatorOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            queries_mask=queries_mask,
            positives_mask=positives_mask,
            negatives_mask=negatives_mask,
            labels=labels,
            teacher_sentence_embeddings=torch.cat(embeddings),
        )
