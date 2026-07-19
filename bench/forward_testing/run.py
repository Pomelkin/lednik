import gc
import json
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import click
import torch
import triton.testing
from kostyl.utils import setup_logger
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from lednik.models import AutoLednikModel, is_lednik_checkpoint, LednikModel
from lednik.models.modeling_lednik import unpad_inputs
from lednik.path_utils import determine_path


logger = setup_logger(fmt="message_only")

type InputMode = Literal["padded", "varlen"]
type ModelInputs = dict[str, torch.Tensor | int]
type ModelType = PreTrainedModel | LednikModel


def _setup_output_file(path: Path) -> None:
    if path.suffix != ".jsonl":
        raise ValueError("Output file must have .jsonl extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    return


def _dump_record(record: dict[str, Any], output_file: Path | None) -> None:
    if output_file is None:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        json_data = json.dumps(record, ensure_ascii=False)
        f.write(f"{json_data}\n")
    return


def _load_model(
    path_name_or_id: str,
    dtype: torch.dtype,
    attn_implementation: str | None,
) -> ModelType:
    """Resolve a ClearML ID / HF repo / local path and load the checkpoint (same as serving)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    kwargs: dict[str, Any] = (
        {}
        if attn_implementation is None
        else {"attn_implementation": attn_implementation}
    )
    path = determine_path(path_name_or_id, is_tokenizer=False)
    if is_lednik_checkpoint(path):
        model = AutoLednikModel.from_pretrained(
            path, weights_prefix="student.", strict_prefix=True, **kwargs
        )
    else:
        model = AutoModel.from_pretrained(path, **kwargs)
    return model.to(device=torch.device("cuda", index=0), dtype=dtype).eval()  # ty:ignore[missing-argument]


def _generate_inputs(
    vocab_size: int,
    pad_token_id: int,
    batch_size: int,
    min_seqlen: int,
    max_seqlen: int,
    seed: int,
) -> tuple[ModelInputs, ModelInputs, int]:
    """Build one shared batch: padded inputs, its varlen view, and the real token count."""
    gen = random.Random(seed)
    torch_gen = torch.Generator().manual_seed(seed)
    lengths = [gen.randint(min_seqlen, max_seqlen) for _ in range(batch_size)]
    samples = [
        torch.randint(0, vocab_size, (length,), dtype=torch.long, generator=torch_gen)
        for length in lengths
    ]

    input_ids = pad_sequence(samples, batch_first=True, padding_value=pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    padded: ModelInputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    varlen = unpad_inputs(input_ids, attention_mask).to_model_inputs()
    return padded, varlen, sum(lengths)


def _bench_model(
    label: str,
    path_name_or_id: str,
    modes: tuple[tuple[InputMode, str | None], ...],
    padded_inputs: ModelInputs,
    varlen_inputs: ModelInputs,
    total_tokens: int,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # reload per mode: the padded path needs eager attention, otherwise
    # Lednik models unpad internally and both modes measure varlen
    for mode, attn_implementation in modes:
        torch.cuda.empty_cache()
        base_vram = torch.cuda.memory_allocated()

        model = _load_model(path_name_or_id, dtype, attn_implementation)
        num_parameters = sum(p.numel() for p in model.parameters())
        model_vram = torch.cuda.memory_allocated() - base_vram
        logger.info(
            f"Loaded '{label}' ({mode}, attn={attn_implementation}): "
            f"{num_parameters / 1e6:.2f}M parameters."
        )

        inputs = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in (varlen_inputs if mode == "varlen" else padded_inputs).items()
        }
        torch.cuda.reset_peak_memory_stats()
        activation_base = torch.cuda.memory_allocated()
        with torch.inference_mode():
            times: list[float] = triton.testing.do_bench(
                lambda model=model, inputs=inputs: model(**inputs),
                warmup=warmup,
                rep=rep,
                return_mode="all",
            )
        activation_vram = torch.cuda.max_memory_allocated() - activation_base

        mean_ms = statistics.fmean(times)
        records.append(
            {
                "Model": label,
                "InputMode": mode,
                "AttnImplementation": attn_implementation,
                "Timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "Dtype": str(dtype).removeprefix("torch."),
                "NumParametersM": round(num_parameters / 1e6, 2),
                "TotalTokens": total_tokens,
                "LatencyMeanMs": round(mean_ms, 4),
                "LatencyMedianMs": round(statistics.median(times), 4),
                "LatencyMinMs": round(min(times), 4),
                "LatencyMaxMs": round(max(times), 4),
                "TokensPerSec": round(total_tokens / (mean_ms / 1000.0), 1),
                "ModelVRAMMb": int(model_vram / 2**20),
                "ActivationVRAMMb": int(activation_vram / 2**20),
            }
        )

        del inputs
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return records


@click.command()
@click.option(
    "--tokenizer",
    "tokenizer_path_name_or_id",
    type=str,
    required=True,
    help="Local path, ClearML model ID or HF Hub repo id of the tokenizer.",
)
@click.option(
    "--teacher",
    type=str,
    default=None,
    help="Teacher model checkpoint; benched with padded inputs (sdpa).",
)
@click.option(
    "--lednik-full-attn",
    type=str,
    default=None,
    help="Full-attention Lednik checkpoint; benched padded (eager) and varlen (flash_attention_2).",
)
@click.option(
    "--lednik-hybrid",
    type=str,
    default=None,
    help="Hybrid Lednik checkpoint; benched with varlen inputs (flash_attention_2).",
)
@click.option(
    "--static",
    "static_model",
    type=str,
    default=None,
    help="Static embeddings checkpoint; benched with padded inputs.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="JSONL file to append metric records to; omit to only log the results.",
)
@click.option(
    "--run-name",
    type=str,
    default=None,
    help="Optional label stored with the records.",
)
@click.option("--batch-size", type=click.IntRange(min=1), default=8, show_default=True)
@click.option(
    "--min-seqlen", type=click.IntRange(min=1), default=128, show_default=True
)
@click.option(
    "--max-seqlen", type=click.IntRange(min=1), default=2048, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--warmup",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="do_bench warmup time per model, ms.",
)
@click.option(
    "--rep",
    type=click.IntRange(min=1),
    default=5000,
    show_default=True,
    help="do_bench measurement time per model, ms.",
)
def run(
    tokenizer_path_name_or_id: str,
    teacher: str | None,
    lednik_full_attn: str | None,
    lednik_hybrid: str | None,
    static_model: str | None,
    output_file: Path | None,
    run_name: str | None,
    batch_size: int,
    min_seqlen: int,
    max_seqlen: int,
    seed: int,
    warmup: int,
    rep: int,
) -> None:
    """Benchmark the pure forward pass of embedding models with triton's do_bench."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if min_seqlen > max_seqlen:
        raise click.UsageError("--min-seqlen must not exceed --max-seqlen.")

    slots: list[tuple[str, str, tuple[tuple[InputMode, str | None], ...]]] = [
        (label, path, modes)
        for label, path, modes in (
            ("teacher", teacher, (("padded", "sdpa"),)),
            (
                "lednik-full-attn",
                lednik_full_attn,
                (("padded", "eager"), ("varlen", "flash_attention_2")),
            ),
            ("lednik-hybrid", lednik_hybrid, (("varlen", "flash_attention_2"),)),
            ("static", static_model, (("padded", None),)),
        )
        if path is not None
    ]
    if not slots:
        raise click.UsageError(
            "Provide at least one of --teacher, --lednik-full-attn, "
            "--lednik-hybrid or --static."
        )

    if output_file is not None:
        _setup_output_file(output_file)

    tokenizer_path = determine_path(tokenizer_path_name_or_id, is_tokenizer=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer from path '{tokenizer_path}'.")

    padded_inputs, varlen_inputs, total_tokens = _generate_inputs(
        vocab_size=int(tokenizer.vocab_size),
        pad_token_id=int(tokenizer.pad_token_id),
        batch_size=batch_size,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
        seed=seed,
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(
        f"Batch: {batch_size} sequences of {min_seqlen}..{max_seqlen} tokens, "
        f"{total_tokens} tokens total, dtype {dtype}."
    )

    base_fields = {"RunName": run_name} if run_name is not None else {}
    base_fields.update(
        {
            "BatchSize": batch_size,
            "MinSeqLen": min_seqlen,
            "MaxSeqLen": max_seqlen,
            "Seed": seed,
        }
    )
    for label, path_name_or_id, modes in slots:
        records = _bench_model(
            label,
            path_name_or_id,
            modes,
            padded_inputs,
            varlen_inputs,
            total_tokens,
            dtype,
            warmup,
            rep,
        )
        for record in records:
            record.update(base_fields)
            _dump_record(record, output_file)
            logger.info(json.dumps(record, ensure_ascii=False, indent=2))
    return


if __name__ == "__main__":
    run()
