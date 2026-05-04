from typing import Literal

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel


@torch.inference_mode()
def extract_embeddings(
    model: PreTrainedModel,
    vocab_tokens: list[list[int]] | list[torch.Tensor],
    pad_token: int,
    pooling: Literal["mean", "last", "cls"],
    batch_size: int,
) -> torch.Tensor:
    """
    Extract embeddings from a pre-trained model for given vocabulary tokens using specified pooling.

    Args:
        model: The pre-trained transformer model to use for embedding extraction.
        vocab_tokens: A list of tokens.
        pad_token: The token ID used for padding sequences to the same length.
        pooling: The pooling strategy to use. "mean" computes the average
            of the hidden states weighted by the attention mask, while "last" uses the last non-padded token's embedding.
        batch_size: The number of sequences to process in each batch.

    Returns:
        A tensor containing the extracted embeddings
            for each input sequence.

    """
    model.eval()

    inputs_l = [
        torch.tensor(tokens, dtype=torch.long)
        if isinstance(tokens, list)
        else tokens.long()
        for tokens in vocab_tokens
    ]

    progress_bar = tqdm(total=len(inputs_l), desc="Extracting embeddings", unit="token")
    embeddings_list = []
    for i in range(0, len(inputs_l), batch_size):
        batch_inputs = inputs_l[i : i + batch_size]

        inputs_t = torch.nn.utils.rnn.pad_sequence(
            batch_inputs,
            batch_first=True,
            padding_value=pad_token,
        ).to(model.device)
        attention_mask = (inputs_t != pad_token).to(model.device, dtype=torch.long)

        outputs = model(input_ids=inputs_t, attention_mask=attention_mask)
        last_hidden_state: torch.Tensor = outputs[0]

        batch_embeddings = get_sentence_embedding(
            last_hidden_state, attention_mask, pooling
        )

        batch_embeddings_list = torch.unbind(batch_embeddings, dim=0)
        embeddings_list.extend(batch_embeddings_list)
        progress_bar.update(len(batch_inputs))
    return torch.stack(embeddings_list, dim=0)


def get_sentence_embedding(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_method: Literal["cls", "mean", "last"],
) -> torch.Tensor:
    """
    Compute sentence embeddings from token embeddings using the specified pooling method.

    Args:
        hidden_states: Tensor of shape (batch_size, seq_length, embedding_dim) containing token embeddings.
        attention_mask: Tensor of shape (batch_size, seq_length) indicating valid tokens.
        pooling_method: The pooling strategy to use. Must be one of "cls", "mean", or "last".

    Returns:
        Tensor of shape (batch_size, embedding_dim) containing the sentence embeddings.

    """
    if pooling_method == "cls":
        return hidden_states[:, 0, :]
    elif pooling_method == "mean":
        masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return sum_embeddings / lengths
    elif pooling_method == "last":
        lengths = (attention_mask.cumsum(dim=-1) - 1).amax(dim=-1)
        return hidden_states[torch.arange(hidden_states.size(0)), lengths]
    else:
        raise ValueError(f"Unsupported pooling method: {pooling_method}")
