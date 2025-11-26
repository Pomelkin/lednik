from string import punctuation
from typing import cast

import torch
from tokenizers import Regex
from tokenizers import Tokenizer
from tokenizers.normalizers import Replace
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import Strip
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerBase


def _get_default_post_processor_template() -> TemplateProcessing:
    return TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[],
    )


def _replace_normalizer(
    tokenizer: Tokenizer,
) -> Tokenizer:
    """
    Replace the normalizer for the tokenizer.

    The new normalizer will replace punctuation with a space before and after the punctuation.
    It will also replace multiple spaces with a single space and strip the right side of the string.
    If the tokenizer already has a normalizer, it will be added to the new normalizer.
    If the tokenizer does not have a normalizer, a new normalizer will be created.

    Args:
        tokenizer: The tokenizer to change.

    Returns:
        The tokenizer with a replaced normalizer.

    """
    spaces_punctuation = tokenizer.encode("a, ,", add_special_tokens=False).tokens
    if len(spaces_punctuation) != 3:
        add_space = False
    else:
        _, first_comma, second_comma = spaces_punctuation
        add_space = first_comma == second_comma == ","

    normalizer = tokenizer.normalizer
    new_normalizers = []
    for char in punctuation:
        replacement = f" {char} " if add_space else f"{char} "
        new_normalizers.append(Replace(char, replacement))

    new_normalizers.append(Replace(Regex(r"\s+"), " "))
    new_normalizers.append(Strip(right=True))
    if normalizer is None:
        normalizer = Sequence(new_normalizers)  # type: ignore
    else:
        normalizer = Sequence([normalizer, *new_normalizers])  # type: ignore
    tokenizer.normalizer = normalizer  # type: ignore
    return tokenizer


def _replace_post_processor(
    tokenizer: Tokenizer,
) -> Tokenizer:
    """Replace the post-processor for the tokenizer."""
    tokenizer.post_processor = _get_default_post_processor_template()  # type: ignore
    return tokenizer


def customize_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """
    Modify the tokenizer by replacing its normalizer and post-processor.

    Args:
        tokenizer: The tokenizer to modify.

    Returns:
        The modified tokenizer.

    """
    backend_tokenizer = tokenizer.backend_tokenizer
    backend_tokenizer = cast(Tokenizer, backend_tokenizer)
    _replace_normalizer(backend_tokenizer)
    _replace_post_processor(backend_tokenizer)
    return tokenizer


def calculate_token_weights(
    vocab_size: int, sif_coefficient: float = 1e-4
) -> torch.Tensor:
    """
    Calculate token weights using the SIF weighting scheme.

    Args:
        vocab_size: The size of the vocabulary.
        sif_coefficient: The SIF coefficient to use in the weighting scheme.

    Returns:
        A tensor of token weights.

    """
    inv_rank = 1 / (torch.arange(2, vocab_size + 2))
    proba = inv_rank / torch.sum(inv_rank)
    weight = sif_coefficient / (sif_coefficient + proba)
    return weight
