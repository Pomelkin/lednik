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
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, OpenAIError
from tqdm import tqdm


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

    expanded_texts = texts * 10
    gen = random.Random(42)
    gen.shuffle(expanded_texts)
    return expanded_texts


async def _send_request(
    client: AsyncOpenAI,
    text: str,
    model: str,
) -> RequestResult:
    start = time.perf_counter()
    try:
        await client.embeddings.create(model=model, input=text, encoding_format="float")
        ok = True
    except OpenAIError:
        ok = False
    return RequestResult(start=start, end=time.perf_counter(), ok=ok)


async def _run_warmup(
    client: AsyncOpenAI,
    texts: list[str],
    model: str,
    num_requests: int,
) -> None:
    warmup_texts = texts[:num_requests]
    progress = tqdm(total=len(warmup_texts), desc="Warmup", leave=False)
    for offset in range(0, len(warmup_texts), WARMUP_CONCURRENCY):
        chunk = warmup_texts[offset : offset + WARMUP_CONCURRENCY]
        await asyncio.gather(*[_send_request(client, text, model) for text in chunk])
        progress.update(len(chunk))
    progress.close()
    return


async def _run_open_loop(
    client: AsyncOpenAI,
    texts: list[str],
    model: str,
    rps: int,
) -> LoadTestResult:
    """Fire one request per 1/rps seconds on an absolute schedule, not waiting for responses."""
    interval = 1.0 / rps
    progress = tqdm(total=len(texts), desc=f"Open-loop (target {rps} rps)")
    tasks: list[asyncio.Task[RequestResult]] = []
    lateness: list[float] = []
    loop_start = time.perf_counter()
    for i, text in enumerate(texts):
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
        task = asyncio.create_task(_send_request(client, text, model))
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
    client: AsyncOpenAI,
    texts: list[str],
    model: str,
    concurrency: int,
) -> LoadTestResult:
    """Send waves of `concurrency` parallel requests, next wave starts after the previous completes."""
    progress = tqdm(total=len(texts), desc=f"Closed-loop (waves of {concurrency})")
    results: list[RequestResult] = []
    request_time = 0.0
    for offset in range(0, len(texts), concurrency):
        chunk = texts[offset : offset + concurrency]
        wave_start = time.perf_counter()
        wave_results = await asyncio.gather(
            *[_send_request(client, text, model) for text in chunk]
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
    texts: list[str],
    model: str,
    rps: int,
    concurrency: int,
    warmup_requests: int,
    timeout: float,
    run_name: str,
    output_file: Path | None,
) -> None:
    # max_retries=0: retries would silently inflate the measured latencies;
    # unlimited connections: the default pool of 100 would queue requests
    # client-side under heavy load and pollute the measured latencies
    async with AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key="required-not-used",
        timeout=timeout,
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            timeout=timeout,
        ),
    ) as client:
        await _run_warmup(client, texts, model, warmup_requests)

        if mode in ("open", "both"):
            load_result = await _run_open_loop(client, texts, model, rps)
            record = _build_record(
                run_name, "open-loop", load_result, target_rps=rps, concurrency=None
            )
            _dump_record(record, output_file)
            logger.info(
                f"Open-loop: {json.dumps(record, ensure_ascii=False, indent=2)}"
            )

        if mode in ("closed", "both"):
            load_result = await _run_closed_loop(client, texts, model, concurrency)
            record = _build_record(
                run_name,
                "closed-loop",
                load_result,
                target_rps=None,
                concurrency=concurrency,
            )
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
    "--model",
    type=str,
    default="lednik",
    show_default=True,
    help="Model name sent in the OpenAI embeddings payload.",
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
    model: str,
) -> None:
    """Measure latency percentiles and throughput of an embedding server."""
    if output_file is not None:
        _setup_output_file(output_file)
    texts = _load_texts(input_path)
    logger.info(f"Loaded {len(texts)} texts from '{input_path}'.")

    asyncio.run(
        _main(
            base_url=f"http://{host}:{port}",
            mode=mode,
            texts=texts,
            model=model,
            rps=rps,
            concurrency=concurrency,
            warmup_requests=warmup_requests,
            timeout=timeout,
            run_name=run_name,
            output_file=output_file,
        )
    )
    return


if __name__ == "__main__":
    run()
