import random
import asyncio
import csv
import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import click
import httpx
from kostyl.utils import setup_logger
from tqdm import tqdm
from transformers import AutoTokenizer

from lednik.path_utils import determine_path


logger = setup_logger(fmt="message_only")

WARMUP_CONCURRENCY = 4


@dataclass
class RequestResult:  # noqa: D101
    start: float
    end: float
    ok: bool

    @property
    def latency(self) -> float:  # noqa: D102
        return self.end - self.start


@dataclass
class LoadTestResult:  # noqa: D101
    results: list[RequestResult]
    request_time: float
    """Accumulated pure request time (no pipeline overhead), seconds."""
    send_jitter_p99_ms: float | None = None
    """Harness self-check: how late open-loop dispatches fired vs the schedule."""


def _p99(sorted_values: list[float]) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) < 2:
        return sorted_values[-1]
    return statistics.quantiles(sorted_values, n=100, method="inclusive")[98]


def _setup_output_file(path: Path) -> None:
    if path.suffix != ".jsonl":
        raise ValueError("Output file must have .jsonl extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    return


def _load_texts(input_path: Path) -> list[str]:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"Input file '{input_path}' must have a 'text' column.")
        texts = [row["text"] for row in reader if row["text"]]
    if not texts:
        raise ValueError(f"Input file '{input_path}' contains no texts.")
    return texts


def _sample_inputs(
    lengths: list[int],
    vocab_ids: list[int],
    num_samples: int,
    scale_factor: float,
    max_seq_length: int | None,
    seed: int = 42,
) -> list[list[int]]:
    """Generate random-id sequences whose lengths follow the scaled production distribution.

    Embedding quality is irrelevant for throughput measurements, so token ids
    are sampled uniformly; only the varlen length profile matters.
    """
    gen = random.Random(seed)
    inputs: list[list[int]] = []
    for _ in range(num_samples):
        length = max(1, round(gen.choice(lengths) * scale_factor))
        if max_seq_length is not None:
            length = min(length, max_seq_length)
        inputs.append(gen.choices(vocab_ids, k=length))
    return inputs


async def _send_request(
    client: httpx.AsyncClient,
    token_ids: list[int],
) -> RequestResult:
    start = time.perf_counter()
    try:
        response = await client.post("/predict", json={"token_ids": token_ids})
        ok = response.status_code == httpx.codes.OK
    except httpx.HTTPError:
        ok = False
    return RequestResult(start=start, end=time.perf_counter(), ok=ok)


async def _run_warmup(
    client: httpx.AsyncClient,
    inputs: list[list[int]],
    num_requests: int,
) -> None:
    warmup_inputs = inputs[:num_requests]
    progress = tqdm(total=len(warmup_inputs), desc="Warmup", leave=False)
    for offset in range(0, len(warmup_inputs), WARMUP_CONCURRENCY):
        chunk = warmup_inputs[offset : offset + WARMUP_CONCURRENCY]
        await asyncio.gather(*[_send_request(client, ids) for ids in chunk])
        progress.update(len(chunk))
    progress.close()
    return


async def _run_open_loop(
    client: httpx.AsyncClient,
    inputs: list[list[int]],
    rps: int,
) -> LoadTestResult:
    """Fire one request per 1/rps seconds on an absolute schedule, not waiting for responses."""
    interval = 1.0 / rps
    progress = tqdm(total=len(inputs), desc=f"Open-loop (target {rps} rps)")
    tasks: list[asyncio.Task[RequestResult]] = []
    lateness: list[float] = []
    loop_start = time.perf_counter()
    for i, token_ids in enumerate(inputs):
        # absolute schedule from a fixed start: one late dispatch
        # doesn't shift the rest and drift doesn't accumulate
        target = loop_start + i * interval
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            # behind schedule: dispatch back-to-back, but yield so the
            # in-flight request tasks are not starved
            await asyncio.sleep(0)
        lateness.append(time.perf_counter() - target)
        task = asyncio.create_task(_send_request(client, token_ids))
        task.add_done_callback(lambda _: progress.update(1))
        tasks.append(task)
    results = list(await asyncio.gather(*tasks))
    progress.close()

    # pure request time: from the first request start to the last response;
    # the scheduling sleeps overlap with in-flight requests, so they are free
    request_time = max(r.end for r in results) - min(r.start for r in results)
    return LoadTestResult(
        results=results,
        request_time=request_time,
        send_jitter_p99_ms=_p99(sorted(late * 1000 for late in lateness)),
    )


async def _run_closed_loop(
    client: httpx.AsyncClient,
    inputs: list[list[int]],
    concurrency: int,
) -> LoadTestResult:
    """Send waves of `concurrency` parallel requests, next wave starts after the previous completes."""
    progress = tqdm(total=len(inputs), desc=f"Closed-loop (waves of {concurrency})")
    results: list[RequestResult] = []
    request_time = 0.0
    for offset in range(0, len(inputs), concurrency):
        chunk = inputs[offset : offset + concurrency]
        wave_start = time.perf_counter()
        wave_results = await asyncio.gather(
            *[_send_request(client, ids) for ids in chunk]
        )
        # accumulate only the in-flight window of each wave,
        # excluding the pipeline overhead between waves
        request_time += max(r.end for r in wave_results) - wave_start
        results.extend(wave_results)
        progress.update(len(chunk))
    progress.close()
    return LoadTestResult(results=results, request_time=request_time)


def _build_record(
    run_name: str,
    mode: str,
    load_result: LoadTestResult,
    target_rps: int | None,
    concurrency: int | None,
) -> dict[str, Any]:
    successful = [r for r in load_result.results if r.ok]
    num_errors = len(load_result.results) - len(successful)
    latencies_ms = sorted(r.latency * 1000 for r in successful)

    record: dict[str, Any] = {
        "RunName": run_name,
        "Mode": mode,
        "Timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "TargetRPS": target_rps,
        "Concurrency": concurrency,
        "NumRequests": len(load_result.results),
        "NumErrors": num_errors,
        "RequestTimeSec": round(load_result.request_time, 4),
        "ThroughputRPS": round(len(successful) / load_result.request_time, 4)
        if load_result.request_time > 0
        else None,
    }
    if load_result.send_jitter_p99_ms is not None:
        # generator lateness vs the schedule; trustworthy run when far below 1000/rps
        record["SendJitterP99Ms"] = round(load_result.send_jitter_p99_ms, 3)

    if len(latencies_ms) >= 2:
        centiles = statistics.quantiles(latencies_ms, n=100, method="inclusive")
        record.update(
            {
                "LatencyMeanMs": round(statistics.fmean(latencies_ms), 3),
                "LatencyP50Ms": round(centiles[49], 3),
                "LatencyP90Ms": round(centiles[89], 3),
                "LatencyP95Ms": round(centiles[94], 3),
                "LatencyP99Ms": round(centiles[98], 3),
                "LatencyMinMs": round(latencies_ms[0], 3),
                "LatencyMaxMs": round(latencies_ms[-1], 3),
            }
        )
    return record


def _dump_record(record: dict[str, Any], output_file: Path | None) -> None:
    if output_file is None:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        json_data = json.dumps(record, ensure_ascii=False)
        f.write(f"{json_data}\n")
    return


async def _main(
    base_url: str,
    mode: Literal["open", "closed", "both"],
    inputs: list[list[int]],
    rps: int,
    concurrency: int,
    warmup_requests: int,
    timeout: float,
    run_name: str,
    output_file: Path | None,
    scale_factor: float,
) -> None:
    mean_tokens = round(statistics.fmean(len(ids) for ids in inputs), 1)
    # unlimited connections: the default pool of 100 would queue requests
    # client-side under heavy load and pollute the measured latencies;
    # trust_env=False: localhost traffic must never go through HTTP(S)_PROXY
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        trust_env=False,
    ) as client:
        await _run_warmup(client, inputs, warmup_requests)

        if mode in ("open", "both"):
            load_result = await _run_open_loop(client, inputs, rps)
            record = _build_record(
                run_name, "open-loop", load_result, target_rps=rps, concurrency=None
            )
            record["ScaleFactor"] = scale_factor
            record["MeanTokensPerRequest"] = mean_tokens
            _dump_record(record, output_file)
            logger.info(
                f"Open-loop: {json.dumps(record, ensure_ascii=False, indent=2)}"
            )

        if mode in ("closed", "both"):
            load_result = await _run_closed_loop(client, inputs, concurrency)
            record = _build_record(
                run_name,
                "closed-loop",
                load_result,
                target_rps=None,
                concurrency=concurrency,
            )
            record["ScaleFactor"] = scale_factor
            record["MeanTokensPerRequest"] = mean_tokens
            _dump_record(record, output_file)
            logger.info(
                f"Closed-loop: {json.dumps(record, ensure_ascii=False, indent=2)}"
            )
    return


@click.command()
@click.option("--host", type=str, default="localhost", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option(
    "--mode",
    type=click.Choice(["open", "closed", "both"]),
    default="both",
    show_default=True,
    help="open: fixed request rate; closed: waves of parallel requests.",
)
@click.option(
    "--rps",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Target requests per second for the open-loop mode.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Wave size (parallel requests) for the closed-loop mode.",
)
@click.option(
    "--input-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(__file__).parent / "inputs.csv",
    show_default=True,
    help="CSV file with a 'text' column.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="JSONL file to append metric records to; omit to only log the results.",
)
@click.option(
    "--run-name", type=str, required=True, help="Label stored with the records."
)
@click.option(
    "--warmup-requests",
    type=click.IntRange(min=0),
    default=32,
    show_default=True,
    help="Requests sent before measuring to warm the server up.",
)
@click.option(
    "--timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Per-request timeout in seconds; timed out requests count as errors.",
)
@click.option(
    "--tokenizer",
    "tokenizer_path_name_or_id",
    type=str,
    required=True,
    help="Local path, ClearML model ID or HF Hub repo id of the tokenizer "
    "(must match the one the server was started with).",
)
@click.option(
    "--max-seq-length",
    type=int,
    default=None,
    show_default=True,
    help="Cap on sampled sequence lengths; should match the server's truncation.",
)
@click.option(
    "--scale-factor",
    type=click.FloatRange(min=0.01),
    default=1.0,
    show_default=True,
    help="Multiplier applied to the production token-length distribution.",
)
@click.option(
    "--num-samples",
    type=click.IntRange(min=1),
    default=5000,
    show_default=True,
    help="Number of synthetic requests generated for the run.",
)
def run(
    host: str,
    port: int,
    mode: Literal["open", "closed", "both"],
    rps: int,
    concurrency: int,
    input_path: Path,
    output_file: Path | None,
    run_name: str,
    warmup_requests: int,
    timeout: float,
    tokenizer_path_name_or_id: str,
    max_seq_length: int | None,
    scale_factor: float,
    num_samples: int,
) -> None:
    """Measure latency percentiles and throughput of an embedding server."""
    if output_file is not None:
        _setup_output_file(output_file)
    texts = _load_texts(input_path)
    logger.info(f"Loaded {len(texts)} texts from '{input_path}'.")

    tokenizer_path = determine_path(tokenizer_path_name_or_id, is_tokenizer=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer from path '{tokenizer_path}'.")

    # texts are only the source of the production length distribution;
    # the actual payloads are synthesized to keep the varlen profile at any scale
    lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
    special_ids = set(tokenizer.all_special_ids)
    vocab_ids = [i for i in range(len(tokenizer)) if i not in special_ids]
    inputs = _sample_inputs(
        lengths,
        vocab_ids,
        num_samples,
        scale_factor,
        max_seq_length or tokenizer.model_max_length,
    )
    sampled_lengths = sorted(len(ids) for ids in inputs)
    logger.info(
        f"Sampled {num_samples} sequences (scale x{scale_factor}): "
        f"mean {statistics.fmean(sampled_lengths):.1f}, "
        f"p50 {sampled_lengths[len(sampled_lengths) // 2]}, "
        f"max {sampled_lengths[-1]} tokens."
    )

    asyncio.run(
        _main(
            base_url=f"http://{host}:{port}",
            mode=mode,
            inputs=inputs,
            rps=rps,
            concurrency=concurrency,
            warmup_requests=warmup_requests,
            timeout=timeout,
            run_name=run_name,
            output_file=output_file,
            scale_factor=scale_factor,
        )
    )
    return


if __name__ == "__main__":
    run()
