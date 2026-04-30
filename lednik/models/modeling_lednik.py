import contextlib
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import cast

import torch
import torch.nn.functional as F
from kostyl.ml.integrations.lightning import LightningCheckpointLoaderMixin
from kostyl.utils import setup_logger
from liger_kernel.transformers import LigerGEGLUMLP
from liger_kernel.transformers import LigerRMSNorm
from liger_kernel.transformers import LigerSwiGLUMLP
from torch import Tensor
from torch import nn
from torch.nn.attention import varlen
from transformers import PreTrainedModel
from transformers.integrations import use_kernel_func_from_hub
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.utils import is_flash_attn_2_available, is_flash_attn_4_available
from transformers.utils.generic import is_flash_attention_requested
from transformers.utils.generic import maybe_autocast
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.modeling_rope_utils import RopeParameters


from .configuration_lednik import LednikConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_qkvpacked_func  # type: ignore[import]
    from flash_attn.ops.triton.rotary import apply_rotary  # type: ignore[import]

if is_flash_attn_4_available():
    from flash_attn.cute import flash_attn_varlen_func  # type: ignore[import]
    from flash_attn.ops.triton.rotary import apply_rotary  # type: ignore[import]

logger = setup_logger()


class LednikRotaryEmbedding(nn.Module):
    """Rotary Position Embedding module."""

    inv_freq: torch.Tensor

    def __init__(
        self, config: LednikConfig, device: torch.device | None = None
    ) -> None:
        """Initializes the Rotary Position Embedding module."""
        super().__init__()
        if config.rope_parameters is None:
            raise ValueError(
                "The `rope_parameters` is None. "
                "Please specify it during the creation of the LednikConfig."
            )
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seqlen = config.max_position_embeddings
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]

        rope_init_fn = (
            ROPE_INIT_FUNCTIONS[self.rope_type]  # type: ignore
            if self.rope_type != "default"
            else self.compute_default_rope_parameters
        )
        inv_freq, scale = rope_init_fn(config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.attention_scaling = scale
        self.is_flash_attn = is_flash_attention_requested(config=config)
        return

    @staticmethod
    def compute_default_rope_parameters(
        config: LednikConfig,
        device: torch.device | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation.

        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).

        """
        dim = getattr(config, "head_dim", None)
        if dim is None:
            if config.num_attention_heads is None:
                raise ValueError(
                    "Cannot calculate head dimension size due to num_attention_heads being None"
                )
            dim = config.hidden_size // config.num_attention_heads

        rope_parameters = cast(RopeParameters, config.rope_parameters)
        base = rope_parameters["rope_theta"]
        if base is None:
            raise ValueError(
                "`rope_theta` must be specified in `rope_parameters` for default RoPE."
            )

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        attention_scale = 1.0  # unused in this type or RoPE
        return inv_freq, attention_scale

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the rotary position embeddings for the given input."""
        inv_freq_unsq = self.inv_freq.unsqueeze(0)
        position_ids_unsq = position_ids.unsqueeze(-1)

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):
            pos_freq = position_ids_unsq.float() @ inv_freq_unsq.float()

            if not self.is_flash_attn:  # for fa apply_rotary cos and sin must have last dim eq to head_dim / 2, not head_dim
                pos_freq = torch.cat([pos_freq, pos_freq], dim=-1)

            cos = pos_freq.cos() * self.attention_scaling
            sin = pos_freq.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)

    def extra_repr(self) -> str:  # noqa: D102
        return f"rope_type={self.rope_type}, max_seq_len_cached={self.max_seq_len_cached}, theta={self.config.rope_parameters['rope_theta']}"  # type: ignore


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(dim=-1, chunks=2)
    return torch.cat([-x2, x1], dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.

    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbUnpad(torch.autograd.Function):
    """Rotary Embedding for padding-free hidden states Autograd Function."""

    @staticmethod
    def forward(  # pyrefly: ignore
        ctx: torch.autograd.function.Function,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for rotary embedding on unpadded sequences.

        Args:
            ctx (`torch.autograd.function.Function`): Context to save information for backward computation.
            q (`torch.Tensor`): Query tensor of shape [total_seq_len, num_heads, head_dim].
            k (`torch.Tensor`): Key tensor of shape [total_seq_len, num_heads, head_dim].
            cos (`torch.Tensor`): Cosine positional embeddings of shape [max_seqlen, head_dim / 2].
            sin (`torch.Tensor`): Sine positional embeddings of shape [max_seqlen, head_dim / 2].
            cu_seqlens (`torch.Tensor`): Cumulative sequence lengths tensor of shape [batch_size + 1].
            max_seqlen (`int`): Maximum sequence length in the batch.

        """
        if cos.ndim != 2:
            raise ValueError(
                f"For padding-free rotary embedding cos must be 2D, but got {cos.ndim}D"
            )
        if sin.ndim != 2:
            raise ValueError(
                f"For padding-free rotary embedding sin must be 2D, but got {sin.ndim}D"
            )
        qk = torch.cat((q, k), dim=1).contiguous()
        apply_rotary(
            x=qk,
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
            interleaved=False,
        )
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen  # type: ignore
        q, k = qk.chunk(dim=1, chunks=2)
        return q, k

    @staticmethod
    def backward(  # pyrefly: ignore
        ctx: torch.autograd.function.Function,
        dq_rotated: torch.Tensor,
        dk_rotated: torch.Tensor,
    ) -> Any:  # ty:ignore[invalid-method-override]
        """Backward pass for rotary embedding on unpadded sequences."""
        dqk_rotated = torch.cat((dq_rotated, dk_rotated), dim=1).contiguous()
        cos, sin, cu_seqlens = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor], ctx.saved_tensors
        )
        apply_rotary(
            x=dqk_rotated,
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,  # type: ignore
            inplace=True,
            interleaved=False,
            conjugate=True,
        )
        dq, dk = dqk_rotated.chunk(dim=1, chunks=2)
        return dq, dk, None, None, None, None


def apply_rotary_pos_emb_unpadded(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,  # must be [max_seqlen, rotary_dim / 2] or [1, max_seqlen, rotary_dim / 2]
    sin: torch.Tensor,  # must be [max_seqlen, rotary_dim / 2] or [1, max_seqlen, rotary_dim / 2]
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to the pad-free query and key tensors.

    Args:
        q (`torch.Tensor`): Query tensor of shape [total_seq_len, num_heads, head_dim].
        k (`torch.Tensor`): Key tensor of shape [total_seq_len, num_heads, head_dim].
        cos (`torch.Tensor`): Cosine positional embeddings of shape [max_seqlen, head_dim / 2] or [1, max_seqlen, head_dim / 2].
        sin (`torch.Tensor`): Sine positional embeddings of shape [max_seqlen, head_dim / 2] or [1, max_seqlen, head_dim / 2].
        cu_seqlens (`torch.Tensor`): Cumulative sequence lengths tensor of shape [batch_size + 1].
        max_seqlen (`int`): Maximum sequence length in the batch.

    """
    if cos.ndim == 3:
        cos = cos.squeeze(0)
    if sin.ndim == 3:
        sin = sin.squeeze(0)
    return RotaryEmbUnpad.apply(q, k, cos, sin, cu_seqlens, max_seqlen)  # type: ignore


def eager_attention_forward(
    module: "LednikAttention",
    q: torch.Tensor,  # [b, seq, num_heads, dim]
    k: torch.Tensor,  # [b, seq, num_heads, dim]
    v: torch.Tensor,  # [b, seq, num_heads, dim]
    attention_mask: torch.Tensor,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention forward function."""
    q = q.transpose(1, 2)  # [b, num_heads, seq, dim]
    k = k.transpose(1, 2)  # [b, num_heads, seq, dim]
    v = v.transpose(1, 2)  # [b, num_heads, seq, dim]
    raw_scores = (q @ k.mT).float()
    raw_scores = raw_scores * module.softmax_scale
    raw_scores = raw_scores + attention_mask
    attention_scores = raw_scores.softmax(dim=-1).to(q.dtype)
    attention_scores = F.dropout(
        attention_scores,
        p=module.config.attention_dropout,
        training=module.training,
    )
    attn_output = attention_scores @ v
    attn_output = attn_output.transpose(1, 2).contiguous()  # [b, seq, num_heads, dim]
    attention_scores = attention_scores.transpose(
        1, 2
    ).contiguous()  # [b, seq, num_heads, seq]
    return attn_output.to(q.dtype), attention_scores


def flash_attention_2_forward(
    module: "LednikAttention",
    q: torch.Tensor,  # [b * seq, num_heads, dim]
    k: torch.Tensor,  # [b * seq, num_heads, dim]
    v: torch.Tensor,  # [b * seq, num_heads, dim]
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash attention forward function."""
    qkv = torch.stack((q, k, v), dim=1)
    original_dtype = qkv.dtype
    if qkv.dtype != module.fa_target_dtype:
        convert_dtype = True
        qkv = qkv.to(module.fa_target_dtype)
    else:
        convert_dtype = False
    attn_output = flash_attn_varlen_qkvpacked_func(
        qkv=qkv,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        softmax_scale=module.softmax_scale,
        dropout_p=module.config.attention_dropout if module.training else 0.0,
    )
    attn_output = cast(Tensor, attn_output)
    if convert_dtype:
        attn_output = attn_output.to(original_dtype)
    return attn_output, torch.empty((1,))  # dummy attention weights


def flash_attention_4_forward(
    module: "LednikAttention",
    q: torch.Tensor,  # [b * seq, num_heads, dim]
    k: torch.Tensor,  # [b * seq, num_heads, dim]
    v: torch.Tensor,  # [b * seq, num_heads, dim]
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash attention forward function."""
    qkv = torch.stack((q, k, v), dim=1)
    original_dtype = qkv.dtype
    if qkv.dtype != module.fa_target_dtype:
        convert_dtype = True
        qkv = qkv.to(module.fa_target_dtype)
    else:
        convert_dtype = False
    attn_output, _ = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_k=max_seqlen,  # type: ignore
        max_seqlen_q=max_seqlen,  # type: ignore
        softmax_scale=module.softmax_scale,
    )
    attn_output = cast(Tensor, attn_output)
    if convert_dtype:
        attn_output = attn_output.to(original_dtype)
    return attn_output, torch.empty((1,))  # dummy attention weights


def torch_varlen_attn_forward(
    module: "LednikAttention",
    q: torch.Tensor,  # [b * seq, num_heads, dim]
    k: torch.Tensor,  # [b * seq, num_heads, dim]
    v: torch.Tensor,  # [b * seq, num_heads, dim]
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash attention forward function."""
    qkv = torch.stack((q, k, v), dim=1)
    original_dtype = qkv.dtype
    if qkv.dtype != module.fa_target_dtype:
        convert_dtype = True
        qkv = qkv.to(module.fa_target_dtype)
    else:
        convert_dtype = False
    attn_output = varlen.varlen_attn(
        query=q,
        key=k,
        value=v,
        cu_seq_q=cu_seqlens,
        cu_seq_k=cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )
    attn_output = cast(Tensor, attn_output)
    if convert_dtype:
        attn_output = attn_output.to(original_dtype)
    return attn_output, torch.empty((1,))  # dummy attention weights


LEDNIK_ATTENTION_FUNCTION = {
    "eager": eager_attention_forward,
    "flash_attention_2": flash_attention_2_forward,
    "flash_attention_4": flash_attention_4_forward,
    "sdpa": torch_varlen_attn_forward,
}


class LednikAttention(nn.Module):
    """Lednik Attention Module. Supports eager and flash attention implementations."""

    def __init__(self, config: LednikConfig) -> None:
        """Initializes the Lednik Attention module."""
        super().__init__()
        if config.num_attention_heads is None:
            raise ValueError(
                "`num_attention_heads` must be provided in config "
                "for attention block initialization"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = LigerRMSNorm(hidden_size=self.head_dim)
        self.k_norm = LigerRMSNorm(hidden_size=self.head_dim)
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.out_dropout = (
            nn.Dropout(config.out_attn_dropout)
            if config.out_attn_dropout > 0
            else nn.Identity()
        )
        self.softmax_scale = self.head_dim**-0.5

        self.is_varlen_attn = (
            is_flash_attention_requested(config=config)
            or config._attn_implementation == "sdpa"
        )
        self.is_flash_attention = is_flash_attention_requested(config=config)
        self.fa_target_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self.config = config
        return

    def forward(
        self,
        hidden_state: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        non_pad_indices: torch.Tensor | None = None,
        seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass of the attention mechanism.

        Args:
            hidden_state (torch.Tensor): Input hidden states.
                Shape: `(batch_size, seq_len, hidden_dim)` (standard) or `(total_seq_len, hidden_dim)` (unpadded).
            cos (torch.Tensor): Cosine component of rotary positional embeddings.
            sin (torch.Tensor): Sine component of rotary positional embeddings.
            attention_mask (torch.Tensor | None, optional): Attention mask for masking out padding
                or future tokens. Defaults to None.
            cu_seqlens (torch.Tensor | None, optional): Cumulative sequence lengths, required
                for unpadded attention implementations (e.g., Flash Attention). Indicates start/end indices
                of sequences in the flattened batch. Defaults to None.
            max_seqlen (int | None, optional): Maximum sequence length in the batch, required
                for unpadded attention implementations. Defaults to None.

            Required for non-Flash Attention unpadded implementations to correctly apply RoPE:

            non_pad_indices (torch.Tensor | None, optional): Indices of non-padding tokens in the flattened batch.
            seqlen (int | None, optional): Sequence length of the input (can be greater than max_seqlen in case of `pad_multiple_of` in collator).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The output hidden states after projection and dropout.
                - Attention weights (dummy for FlashAttention).

        Raises:
            ValueError: If the attention implementation is unpadded but `cu_seqlens` or
                `max_seqlen` are not provided.

        """
        q: torch.Tensor = self.q_proj(hidden_state)
        k: torch.Tensor = self.k_proj(hidden_state)
        v: torch.Tensor = self.v_proj(hidden_state)

        if self.is_varlen_attn:
            if cu_seqlens is None or max_seqlen is None:
                raise ValueError(
                    f"cu_seqlens and max_seqlen must be provided for {self.config._attn_implementation}."
                )
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
            k = self.k_norm(k.view(-1, self.num_heads, self.head_dim))
            v = v.view(-1, self.num_heads, self.head_dim)
        else:
            bs, seq_len, _ = hidden_state.size()
            q = self.q_norm(q.view(bs, seq_len, self.num_heads, self.head_dim))
            k = self.k_norm(k.view(bs, seq_len, self.num_heads, self.head_dim))
            v = v.view(bs, seq_len, self.num_heads, self.head_dim)

        q, k = self._apply_rope(
            q,
            k,
            cos,
            sin,
            non_pad_indices,
            cu_seqlens,
            max_seqlen,
            seqlen,
        )

        attn_output, attention_weights = LEDNIK_ATTENTION_FUNCTION[
            self.config._attn_implementation  # pyrefly: ignore
        ](
            q=q,
            k=k,
            v=v,
            module=self,
            attention_mask=attention_mask,  # type: ignore
            cu_seqlens=cu_seqlens,  # type: ignore
            max_seqlen=max_seqlen,  # type: ignore
        )
        hidden_state = attn_output.view_as(hidden_state)
        hidden_state = self.out_dropout(self.out_proj(hidden_state))
        return (hidden_state, attention_weights)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        non_pad_indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_varlen_attn:
            if cu_seqlens is None:
                raise ValueError(
                    f"cu_seqlens and max_seqlen must be provided for {self.config._attn_implementation}."
                )
            if self.is_flash_attention:
                if max_seqlen is None:
                    raise ValueError(
                        f"cu_seqlens and max_seqlen must be provided for {self.config._attn_implementation}."
                    )
                q, k = apply_rotary_pos_emb_unpadded(
                    q=q,
                    k=k,
                    cos=cos,
                    sin=sin,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            else:
                if non_pad_indices is None or seqlen is None:
                    raise ValueError(
                        "non_pad_indices and seqlen must be provided for torch varlen attention"
                    )
                buf_q = q.new_zeros(
                    ((cu_seqlens.size(0) - 1) * seqlen, self.num_heads, self.head_dim)
                )
                buf_k = k.new_zeros(
                    ((cu_seqlens.size(0) - 1) * seqlen, self.num_heads, self.head_dim)
                )
                buf_q[non_pad_indices] = q
                buf_k[non_pad_indices] = k
                buf_q = buf_q.view(
                    cu_seqlens.size(0) - 1, seqlen, self.num_heads, self.head_dim
                )
                buf_k = buf_k.view(
                    cu_seqlens.size(0) - 1, seqlen, self.num_heads, self.head_dim
                )
                buf_q, buf_k = apply_rotary_pos_emb(
                    q=buf_q,
                    k=buf_k,
                    cos=cos,
                    sin=sin,
                    unsqueeze_dim=2,
                )
                q = buf_q.view(-1, self.num_heads, self.head_dim)[non_pad_indices]
                k = buf_k.view(-1, self.num_heads, self.head_dim)[non_pad_indices]
        else:
            q, k = apply_rotary_pos_emb(
                q=q,
                k=k,
                cos=cos,
                sin=sin,
                unsqueeze_dim=2,
            )
        return q, k


ACT2MLP = {"silu": LigerSwiGLUMLP, "gelu": LigerGEGLUMLP}


class LednikEncoderLayer(GradientCheckpointingLayer):
    """Lednik Encoder Layer that combines attention and MLP blocks with RMSNorm."""

    def __init__(self, config: LednikConfig) -> None:
        """Initializes the Lednik Encoder Layer."""
        super().__init__()
        self.atten_norm = LigerRMSNorm(hidden_size=config.hidden_size)
        self.attention = LednikAttention(config)
        self.mlp_norm = LigerRMSNorm(hidden_size=config.hidden_size)
        self.mlp = ACT2MLP[config.hidden_act](config)
        self.mlp_dropout = (
            nn.Dropout(config.mlp_dropout)
            if config.mlp_dropout > 0.0
            else nn.Identity()
        )
        return

    @property
    def attention_implementation(self) -> str:
        """Returns the attention implementation used in this layer."""
        return self.attention.config._attn_implementation  # pyrefly: ignore

    def forward(
        self,
        hidden_state: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        non_pad_indices: torch.Tensor | None = None,
        seqlen: int | None = None,
    ) -> tuple[torch.Tensor]:
        """
        Performs the forward pass of the Lednik Encoder Layer.

        Required for non-Flash Attention unpadded implementations to correctly apply RoPE:
        Args:
            non_pad_indices (torch.Tensor | None, optional): Indices of non-padding tokens in the flattened batch.
            seqlen (int | None, optional): Sequence length of the input (can be greater than max_seqlen in case of `pad_multiple_of` in collator).
        """
        attn_output, _ = self.attention(
            self.atten_norm(hidden_state),
            cos,
            sin,
            attention_mask,
            cu_seqlens,
            max_seqlen,
            non_pad_indices,
            seqlen,
        )
        hidden_state = hidden_state + attn_output
        mlp_output = self.mlp(self.mlp_norm(hidden_state))
        hidden_state = hidden_state + self.mlp_dropout(mlp_output)
        return hidden_state


class LednikEmbeddings(nn.Module):
    """Lednik Embeddings Module."""

    def __init__(self, config: LednikConfig) -> None:
        """Initializes the Lednik Embeddings module."""
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.emb = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.dropout = (
            nn.Dropout(p=config.embeddings_dropout)
            if config.embeddings_dropout > 0.0
            else nn.Identity()
        )
        self.emb_norm = LigerRMSNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs the forward pass of the Lednik Embeddings module."""
        if input_ids is not None:
            hidden_states = self.emb(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        return self.dropout(self.emb_norm(hidden_states))


class LednikPreTrainedModel(
    PreTrainedModel,
    LightningCheckpointLoaderMixin,  # pyrefly: ignore
):
    """Lednik PreTrained Model class."""

    config_class = LednikConfig
    base_model_prefix = "model"
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_sdpa = True

    _can_record_outputs = {  # noqa: RUF012
        "hidden_states": LednikEncoderLayer,
        "attentions": LednikAttention,
    }

    def _check_and_adjust_attn_implementation(
        self,
        attn_implementation: str | None,
        is_init_check: bool = False,
        allow_all_kernels: bool = False,
    ) -> str:
        if attn_implementation is None:
            with contextlib.suppress(ValueError, ImportError):
                if is_flash_attn_4_available():
                    attn_implementation = "flash_attention_4"
                elif is_flash_attn_2_available():
                    attn_implementation = "flash_attention_2"
                else:
                    attn_implementation = None
        return super()._check_and_adjust_attn_implementation(
            attn_implementation, is_init_check, allow_all_kernels
        )


UnpaddedInputsTuple = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor | None,
    torch.Tensor | None,
]


@dataclass
class UnpaddedInputs:
    """Dataclass for unpadded inputs used in attention forward pass."""

    unpadded_inputs: torch.Tensor
    non_pad_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    unpadded_position_ids: torch.Tensor | None
    unpadded_labels: torch.Tensor | None

    def as_tuple(
        self,
    ) -> UnpaddedInputsTuple:
        """Returns the unpadded inputs as a tuple."""
        return (
            self.unpadded_inputs,
            self.non_pad_indices,
            self.cu_seqlens,
            self.max_seqlen,
            self.unpadded_position_ids,
            self.unpadded_labels,
        )

    def to_model_inputs(
        self, unpadded_inputs_type: Literal["input_ids", "inputs_embeds"] = "input_ids"
    ) -> dict[str, torch.Tensor | int | None]:
        """Formats the unpadded inputs into a dictionary suitable for model input."""
        return {
            unpadded_inputs_type: self.unpadded_inputs,
            "cu_seqlens": self.cu_seqlens,
            "max_seqlen": self.max_seqlen,
            "non_pad_indices": self.non_pad_indices,
            "position_ids": self.unpadded_position_ids,
            "labels": self.unpadded_labels,
        }


def unpad_inputs(
    padded_inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> UnpaddedInputs:
    """
    Removes padding from inputs and simultaneously filters corresponding elements from position_ids and labels.

    Args:
        padded_inputs (torch.Tensor): The input tensor with padding. Can be 2D (batch_size, seq_len)
            or 3D+ (batch_size, seq_len, hidden_dim, ...).
        attention_mask (torch.Tensor): A 2D binary tensor (batch_size, seq_len) where 1 indicates
            valid tokens and 0 indicates padding.
        position_ids (torch.Tensor | None, optional): A 2D tensor (1, seq_len) or 1D tensor of position indices.
             If provided, elements corresponding to padding are removed. Defaults to None.
        labels (torch.Tensor | None, optional): A 2D tensor (batch_size, seq_len) of labels (e.g. for NER).
             If provided, elements corresponding to padding are removed. Defaults to None.

    Returns:
        UnpaddedInputs:
            A tuple containing
            - **unpadded_inputs:** The flattened inputs containing only non-pad tokens.
            - **non_pad_indices:** Indices in the flattened original tensor that correspond to non-pad tokens.
            - **cu_seqlens:** Cumulative sequence lengths (offsets) for the unpadded sequence (batch_size + 1).
            - **max_seqlen:** The maximum sequence length found in the batch.
            - **unpadded_position_ids:** The flattened position IDs for valid tokens, or None.
            - **unpadded_labels:** The flattened labels for valid tokens, or None.

    """
    if attention_mask.ndim != 2:
        raise ValueError(
            f"For unpadding attention_mask must be 2D, got {attention_mask.ndim}D"
        )

    non_pad_indices = attention_mask.flatten().nonzero(as_tuple=False).flatten()
    if padded_inputs.ndim == 2:
        batch_size, _ = padded_inputs.shape
        padded_inputs = padded_inputs.flatten()
    else:
        batch_size, _, *rest_dims = padded_inputs.shape
        padded_inputs = padded_inputs.view(-1, *rest_dims)

    unpadded_inputs = padded_inputs[non_pad_indices]

    seq_lens_in_batch = attention_mask.sum(-1)
    max_seqlen = int(seq_lens_in_batch.max().item())
    cu_seqlens = seq_lens_in_batch.new_zeros((batch_size + 1,))
    cu_seqlens[1:] = seq_lens_in_batch.cumsum(dim=-1)
    cu_seqlens = cu_seqlens.to(torch.int32)

    if position_ids is not None:
        if position_ids.ndim > 2:
            raise ValueError(
                f"For unpadding position_ids must be 1D or 2D, got {position_ids.ndim}D"
            )
        if position_ids.ndim == 2 and position_ids.size(0) != 1:
            raise ValueError(
                f"If position_ids is 2D, it must have batch size of 1, got {position_ids.size(0)}"
            )
        unpadded_position_ids = position_ids.flatten()[non_pad_indices]
    else:
        unpadded_position_ids = None

    if labels is not None:
        if labels.ndim != 2:
            raise ValueError(f"For unpadding labels must be 2D, got {labels.ndim}D")
        unpadded_labels = labels.flatten()[non_pad_indices]
    else:
        unpadded_labels = None

    return UnpaddedInputs(
        unpadded_inputs=unpadded_inputs,
        non_pad_indices=non_pad_indices,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        unpadded_position_ids=unpadded_position_ids,
        unpadded_labels=unpadded_labels,
    )


def pad_outputs(
    unpadded_outputs: torch.Tensor,
    non_pad_indices: torch.Tensor,
    seqlen: int,
    batch_size: int,
    padding_value: int = 0,
) -> torch.Tensor:
    """
    Pads the outputs of a model back to a dense tensor format.

    Args:
        unpadded_outputs (torch.Tensor): A tensor containing only the non-padding elements.
            Can be 1D or multi-dimensional (e.g., [num_valid_tokens, hidden_dim]).
        non_pad_indices (torch.Tensor): A 1D tensor of indices indicating where the
            unpadded elements should be placed in the flattened dense representation.
        seqlen (int): The maximum sequence length of the batch.
        batch_size (int): The number of sequences in the batch.
        padding_value (int, *optional*, defaults to 0): The constant value to use for padding.

    Returns:
        torch.Tensor: A dense tensor of shape (batch_size, seqlen) or
        (batch_size, seqlen, ...) containing the padded outputs.

    """
    if unpadded_outputs.ndim == 1:
        padded_outputs = unpadded_outputs.new_full(
            (batch_size * seqlen,), padding_value
        )
        padded_outputs[non_pad_indices] = unpadded_outputs
        padded_outputs = padded_outputs.view(batch_size, seqlen)
    else:
        _, *rest = unpadded_outputs.shape
        padded_outputs = unpadded_outputs.new_full(
            (batch_size * seqlen, *rest), padding_value
        )
        padded_outputs[non_pad_indices] = unpadded_outputs
        padded_outputs = padded_outputs.view(batch_size, seqlen, *rest)
    return padded_outputs


class LednikModel(LednikPreTrainedModel):
    """Lednik Model."""

    def __init__(self, config: LednikConfig) -> None:
        """Initializes the Lednik Model."""
        super().__init__(config)
        self.embeddings = LednikEmbeddings(config)
        self.rotary_emb = LednikRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [LednikEncoderLayer(config=config) for _ in range(config.num_hidden_layers)]
        )
        self.output_projection = (
            nn.Linear(config.hidden_size, config.output_hidden_size)
            if config.output_hidden_size is not None
            else nn.Identity()
        )
        self.final_norm = LigerRMSNorm(config.output_hidden_size or config.hidden_size)
        self.config = config
        self.is_varlen_attn = (
            is_flash_attention_requested(config=config)
            or config._attn_implementation == "sdpa"
        )
        self.is_flash_attn = is_flash_attention_requested(config=config)
        self.post_init()
        return

    def replace_embeddings(self, new_embeddings: torch.Tensor | nn.Parameter) -> None:
        """Replace the current model embeddings weights with given one."""
        if new_embeddings.numel() != self.embeddings.emb.weight.numel():
            raise ValueError(
                f"new_embeddings should have numel {self.embeddings.emb.weight.numel()}, "
                f"but got {new_embeddings.numel()}."
            )
        if new_embeddings.size() != self.embeddings.emb.weight.size():
            raise ValueError(
                f"new_embeddings should have size {self.embeddings.emb.weight.size()}, "
                f"but got {new_embeddings.size()}."
            )

        new_embeddings = new_embeddings.clone()  # use clone() to avoid weight tying
        if isinstance(new_embeddings, torch.Tensor):
            new_embeddings = nn.Parameter(new_embeddings)
        self.embeddings.emb.weight = new_embeddings
        return

    @merge_with_config_defaults
    @capture_outputs
    def forward(  # noqa: C901
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        non_pad_indices: torch.Tensor | None = None,
        seqlen: int | None = None,
        output_attentions: bool | None = None,
        **_kwargs: Any,
    ) -> BaseModelOutput:
        """
        Forward pass of the Lednik Model.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using `AutoTokenizer`.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range `[0, config.max_position_embeddings - 1]`.
            cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*): Cumulative sequence lengths
                of the input sequences. Used to index the unpadded tensors.
            max_seqlen (`int`, *optional*): Maximum sequence length in the batch
                excluding padding tokens. Used to unpad input_ids and pad output tensors.
            non_pad_indices (`torch.Tensor` of shape `(num_non_pad_tokens,)`, *optional*):
                Indices of non-pad tokens in the flattened input tensor.
                Used to index the unpadded tensors and to pad the output tensors back to the original shape.
            seqlen (`int`, *optional*): The length of each sequence in the batch. Used for correct RoPE application in non-Flash Attention unpadded implementations when `pad_multiple_of` is used in the collator.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.

        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if not (
            (cu_seqlens is None) == (max_seqlen is None) == (non_pad_indices is None)
        ):
            raise ValueError(
                "cu_seqlens, max_seqlen and non_pad_indices either must be provided together or all be None."
            )
        if (
            (cu_seqlens is not None)
            and (max_seqlen is not None)
            and (non_pad_indices is not None)
            and not self.is_varlen_attn
        ):
            logger.warning(
                "cu_seqlens, max_seqlen and non_pad_indices are used only "
                "when attn_implementation is Flash Attention."
            )

        if (input_ids is not None) and (cu_seqlens is None):
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        self._warn_if_output_attentions_and_flash_attn(output_attentions)
        REPAD = False
        if self.is_varlen_attn:  ### VARLEN BRANCH
            # unpad inputs
            if (
                (cu_seqlens is None)
                or (max_seqlen is None)
                or (non_pad_indices is None)
            ):
                if attention_mask is None:
                    raise ValueError(
                        "Attention mask must be provided "
                        "when cu_seqlens, max_seqlen and non_pad_indices are not specified "
                        f"and `attn_implementation` is {self.config._attn_implementation}."
                    )

                if seqlen is None:
                    seqlen = (
                        input_ids.size(1)
                        if input_ids is not None
                        else cast(torch.Tensor, inputs_embeds).size(1)
                    )

                REPAD = True
                if inputs_embeds is None:
                    input_ids = cast(torch.LongTensor, input_ids)
                    if input_ids.ndim != 2:
                        raise ValueError(
                            f"input_ids must be 2D when unpadding, got {input_ids.ndim}D"
                        )
                    with torch.no_grad():
                        output = unpad_inputs(
                            input_ids,
                            attention_mask,
                            position_ids,
                        )
                        (
                            input_ids,
                            non_pad_indices,
                            cu_seqlens,
                            max_seqlen,
                            position_ids,
                            _,
                        ) = output.as_tuple()
                else:
                    if inputs_embeds.ndim < 3:
                        raise ValueError(
                            f"inputs_embeds must be at least 3D when unpadding, got {inputs_embeds.ndim}D"
                        )
                    output = unpad_inputs(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                    )
                    (
                        inputs_embeds,
                        non_pad_indices,
                        cu_seqlens,
                        max_seqlen,
                        position_ids,
                        _,
                    ) = output.as_tuple()
            elif seqlen is None:
                seqlen = max_seqlen

            cu_seqlens = cu_seqlens.to(torch.int32)
        else:  ### PADDED BRANCH
            seqlen = (
                inputs_embeds.size(1)
                if inputs_embeds is not None
                else cast(torch.Tensor, input_ids).size(1)
            )

        if position_ids is None:
            if self.is_flash_attn:
                max_len = cast(int, max_seqlen)
                position_ids = torch.arange(max_len, device=self.device).unsqueeze(0)
            else:
                position_ids = torch.arange(seqlen, device=self.device).unsqueeze(0)

        hidden_state: torch.Tensor = (
            self.embeddings(input_ids=input_ids)
            if inputs_embeds is None
            else self.embeddings(inputs_embeds=inputs_embeds)
        )

        if attention_mask is not None:
            mask = self._prepare_4d_mask(
                hidden_state,
                attention_mask=attention_mask,
            )
        else:
            mask = None

        cos, sin = self.rotary_emb(hidden_state, position_ids)

        for layer in self.layers:
            hidden_state = layer(
                hidden_state=hidden_state,
                cos=cos,
                sin=sin,
                attention_mask=mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                non_pad_indices=non_pad_indices,
                seqlen=seqlen,
            )
        hidden_state = self.output_projection(hidden_state)
        hidden_state = self.final_norm(hidden_state)

        if REPAD:
            if (cu_seqlens is None) or (seqlen is None) or (non_pad_indices is None):
                raise ValueError(
                    "batch_size, seqlen, and non_pad_indices must be defined for repadding."
                )
            hidden_state = pad_outputs(
                hidden_state,
                non_pad_indices,
                seqlen,
                batch_size=cu_seqlens.size(0) - 1,
                padding_value=0,
            )
        return BaseModelOutput(last_hidden_state=hidden_state)

    def _warn_if_output_attentions_and_flash_attn(
        self,
        output_attentions: bool | None = None,
    ) -> None:
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_attentions and is_flash_attention_requested(self.config):
            logger.warning_once(
                "Output_attentions supported only in `eager` mode. "
                "Attention weights will be dummy !"
            )
        return

    def _prepare_4d_mask(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None | torch.Tensor:
        bsz, seqlen = attention_mask.shape
        expanded_mask = attention_mask.reshape(bsz, 1, 1, seqlen).to(hidden_state.dtype)
        bidirectional_mask = (1 - expanded_mask) * float("-inf")
        return bidirectional_mask
