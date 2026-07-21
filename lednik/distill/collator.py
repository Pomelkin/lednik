import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
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
    max_aug_attempts: int = 3

    backend_tokenizer: Tokenizer | None = None

    _bad_texts: set[str] = field(default_factory=set, init=False, repr=False)

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

    @staticmethod
    def _normalize_nested_list(
        data: Any, scalar_type: type, colname: str
    ) -> list[list[Any]]:
        """Normalize a flat list of scalars or a list of lists of scalars to 2D form."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], scalar_type):
                return [data]
            if (
                isinstance(data[0], list)
                and len(data[0]) > 0
                and isinstance(data[0][0], scalar_type)
            ):
                return data
        raise ValueError(
            f"Expected a list of {scalar_type.__name__}s or a non-empty list of lists of "
            f"{scalar_type.__name__}s for column '{colname}', got {type(data)}. Data: {data!r:.300}"
        )

    @staticmethod
    def _normalize_list(data: Any, scalar_type: type, colname: str) -> list[Any]:
        """Normalize a flat list of scalars to 1D form."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], scalar_type):
                return data
        elif isinstance(data, scalar_type):
            return [data]
        raise ValueError(
            f"Expected a {scalar_type.__name__} or a list of {scalar_type.__name__}s for column '{colname}', got {type(data)}. Data: {data!r:.300}"
        )

    @staticmethod
    def _sanity_check(tokens: list[int], embedding: list[float], text: str) -> None:
        if len(tokens) == 0:
            raise ValueError(f"Tokens list is empty for text: {text!r:.300}")
        if len(embedding) == 0:
            raise ValueError(f"Teacher embedding list is empty for text: {text!r:.300}")
        if len(text) == 0:
            raise ValueError("Text is empty.")

        if not isinstance(tokens, list) or not isinstance(tokens[0], int):
            raise ValueError(
                f"Tokens must be a list of ints, got {type(tokens)}. Tokens: {tokens!r:.300}"
            )
        if not isinstance(embedding, list) or not isinstance(embedding[0], float):
            raise ValueError(
                f"Teacher embeddings must be a list of floats, got {type(embedding)}. Embeddings: {embedding!r:.300}"
            )
        if not isinstance(text, str):
            raise ValueError(
                f"Text must be a string, got {type(text)}. Text: {text!r:.300}"
            )
        return

    def _gather_tokens_and_teacher_embeddings(  # noqa: C901
        self,
        items: list[dict[str, Any]],
        text_colname: str,
        tokens_colname: str,
        teacher_emb_colname: str,
    ) -> tuple[list[Tensor], list[Tensor]]:
        texts: list[str] = []
        tokens: list[list[int] | None] = []
        embeddings_l: list[Tensor] = []

        aug_indices_set: set[int] = (
            set(
                random.sample(
                    range(len(items)),
                    k=min(math.ceil(len(items) * self.aug_prob), len(items)),
                )
            )
            if self.aug_prob > 0.0
            else set()
        )

        encode_indices: list[int] = []
        for i, item in enumerate(items):
            try:
                tokens_list: list[list[int]] = self._normalize_nested_list(
                    item[tokens_colname], int, tokens_colname
                )
                teacher_embeddings_list: list[list[float]] = (
                    self._normalize_nested_list(
                        item[teacher_emb_colname], float, teacher_emb_colname
                    )
                )
                text_list: list[str] = self._normalize_list(
                    item[text_colname], str, text_colname
                )
            except KeyError as e:
                raise KeyError(
                    f"Missing expected column in item {i}: {e}. Item keys: {list(item.keys())}"
                ) from e

            if len(tokens_list) != len(teacher_embeddings_list):
                raise ValueError(
                    f"Length mismatch between tokens and teacher embeddings for item {i}. "
                    f"Tokens length: {len(tokens_list)}, Teacher embeddings length: {len(teacher_embeddings_list)}."
                )
            if len(teacher_embeddings_list) != len(text_list):
                raise ValueError(
                    f"Length mismatch between teacher embeddings and texts for item {i}. "
                    f"Teacher embeddings length: {len(teacher_embeddings_list)}, Texts length: {len(text_list)}."
                )

            index = (
                random.randint(0, len(tokens_list) - 1) if len(tokens_list) > 1 else 0
            )

            item_tokens = tokens_list[index]
            item_teacher_embedding = teacher_embeddings_list[index]
            item_text = text_list[index]

            self._sanity_check(item_tokens, item_teacher_embedding, item_text)

            if i in aug_indices_set and item_text not in self._bad_texts:
                if self.corruptor is None:
                    raise ValueError(
                        "Corruptor must be provided for data augmentation."
                    )

                attempt = 0
                while attempt < self.max_aug_attempts:
                    try:
                        corrupted_text = self.corruptor.corrupt(item_text)
                        break
                    except Exception:
                        attempt += 1
                else:
                    logger.warning(
                        f"Failed to corrupt text {item_text} after {self.max_aug_attempts} attempts."
                        " Using original text for tokenization and adding this text to bad set (preventing future aug attempts)."
                    )
                    # Fallback to original text if corruption fails
                    corrupted_text = item_text
                    # Adding this text to a set to avoid future augmentation attempts
                    self._bad_texts.add(item_text)

                encode_indices.append(i)
                texts.append(corrupted_text)
                tokens.append(None)
            else:
                tokens.append(item_tokens)

            item_teacher_embedding_t = torch.tensor(
                item_teacher_embedding, dtype=torch.float32
            )
            if item_teacher_embedding_t.ndim != 1:
                raise ValueError(
                    f"Expected 1D tensor for teacher embedding, got {item_teacher_embedding_t.ndim}D tensor."
                )

            embeddings_l.append(item_teacher_embedding_t)

        if len(texts) > 0:
            encoded_texts = self._encode_texts(texts)
            for idx, enc in zip(encode_indices, encoded_texts, strict=True):
                tokens[idx] = enc

        tokens_l: list[Tensor] = []
        for tok in tokens:
            if tok is None:
                raise RuntimeError(
                    "Some items are still None. Please check the augmentation logic."
                )

            tok = self._postprocess_tokens(tok)  # noqa: PLW2901
            tokens_l.append(torch.tensor(tok, dtype=torch.long))
        return tokens_l, embeddings_l

    def __call__(self, batch: list[dict[str, Any]]) -> CollatorOutput:
        """Collate items into padded inputs, section masks, and teacher embeddings."""
        sequences, embeddings = self._gather_tokens_and_teacher_embeddings(
            batch,
            text_colname=self.query_text_colname,
            tokens_colname=self.query_tok_colname,
            teacher_emb_colname=self.query_teacher_embedding_colname,
        )
        pos_sequences, pos_embeddings = self._gather_tokens_and_teacher_embeddings(
            batch,
            text_colname=self.pos_text_colname,
            tokens_colname=self.pos_tok_colname,
            teacher_emb_colname=self.pos_teacher_embedding_colname,
        )
        sequences += pos_sequences
        embeddings += pos_embeddings

        use_negatives = False
        if (
            self.neg_tok_colname is not None
            and self.neg_text_colname is not None
            and self.neg_teacher_embedding_colname is not None
        ):
            use_negatives = True
            neg_sequences, neg_embeddings = self._gather_tokens_and_teacher_embeddings(
                batch,
                text_colname=self.neg_text_colname,
                tokens_colname=self.neg_tok_colname,
                teacher_emb_colname=self.neg_teacher_embedding_colname,
            )
            sequences += neg_sequences
            embeddings += neg_embeddings

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
            teacher_sentence_embeddings=torch.stack(embeddings),
        )
