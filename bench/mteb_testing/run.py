import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import mteb
from clearml import InputModel
from clearml import Task
from kostyl.ml.integrations.clearml import load_tokenizer_from_clearml
from kostyl.utils import setup_logger
from mteb.get_tasks import MTEBTasks
from mteb.results.model_result import ModelResult

from lednik.models import AutoLednikModel
from mteb_testing.model_wrapper import MTEBModelWrapper


logger = setup_logger()

too_big_bench = {
    "MrTidyRetrieval",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MIRACLRetrievalHardNegatives.v2",
    "MultiLongDocRetrieval",
    "NeuCLIR2022Retrieval",
    "NeuCLIR2022RetrievalHardNegatives",
    "NeuCLIR2023Retrieval",
    "NeuCLIR2023RetrievalHardNegatives",
    "SIB200Classification.v2",
    "EmotionAnalysisPlus",
}


@dataclass
class MTEBRunScores:  # noqa: D101
    scores: dict[str, float]
    avg_score_per_task: dict[str, float]
    avg_score: float
    avg_score_task: float


def _setup_output_file(path: Path) -> None:
    if path.suffix != ".jsonl":
        raise ValueError("Output file must have .jsonl extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    return


def _calculate_scores(result: ModelResult) -> MTEBRunScores:
    task_type2metric: dict[str, dict[str, float]] = defaultdict(dict)
    task_type2sum: dict[str, float] = defaultdict(float)
    for task_result in result.task_results:
        task_type2metric[task_result.task_type][task_result.task_name] = (
            task_result.main_score
        )
        task_type2sum[task_result.task_type] += task_result.main_score

    avg_score_per_task = {
        task_type: task_type2sum[task_type] / len(metrics)
        for task_type, metrics in task_type2metric.items()
    }
    avg_score = sum(task_type2sum.values()) / sum(
        len(metrics) for metrics in task_type2metric.values()
    )
    scores = {
        f"{task_type}/{task_name}": score
        for task_type, metrics in task_type2metric.items()
        for task_name, score in metrics.items()
    }
    return MTEBRunScores(
        scores=scores,
        avg_score_per_task=avg_score_per_task,
        avg_score_task=sum(avg_score_per_task.values()) / len(avg_score_per_task),
        avg_score=avg_score,
    )


def _dump_result(
    avg_score_per_task: dict[str, float],
    avg_score: float,
    avg_score_task: float,
    model_name: str,
    model_revision: str | None,
    output_file: Path,
    task_id: str,
    num_parameters: int | None,
) -> None:
    data2log: dict[str, Any] = {
        "TaskID": task_id,
        "ModelName": model_name,
        "ModelRevision": model_revision,
        "NumParameters": f"{num_parameters / 1_000_000:.2f}M"
        if num_parameters is not None
        else None,
        "AvgScore": avg_score,
        "AvgScoreTask": avg_score_task,
    }
    data2log.update(avg_score_per_task)
    with output_file.open("a", encoding="utf-8") as f:
        json_data = json.dumps(data2log, ensure_ascii=False)
        f.write(f"{json_data}\n")
    return


def _log_result(
    avg_score: float,
    avg_score_task: float,
    avg_score_per_task: dict[str, float],
    task: Task,
    clearml_model: InputModel | None = None,
) -> None:
    task_logger = task.get_logger()
    scores2report = copy.deepcopy(avg_score_per_task)
    scores2report.update({"AvgScore": avg_score, "AvgScoreTask": avg_score_task})

    for name, score in scores2report.items():
        task_logger.report_single_value(name=name, value=score)
        if clearml_model is not None and not clearml_model.published:
            clearml_model.report_single_value(name=name, value=score)

    logger.info(f"Avg Score: {avg_score:.4f}")
    return


def main(
    model_id: str,
    is_clearml_model: bool,
    output_file: Path,
    tokenizer_id: str | None = None,
    max_sequence_length: int | None = None,
    batch_size: int = 128,
) -> None:
    """Main function to run MTEB benchmark on Russian text tasks."""
    task: Task = Task.init(
        project_name="Lednik",
        task_name="MTEB Benchmark Run",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
        auto_connect_frameworks={
            "pytorch": False,
            "tensorboard": True,
            "matplotlib": True,
            "detect_repository": True,
        },
    )

    model = None
    if is_clearml_model:
        if tokenizer_id is None:
            raise ValueError(
                "tokenizer_id must be provided when is_clearml_model is True"
            )
        clearml_model = InputModel(model_id=model_id)
        clearml_model.connect(
            task=task, name="Model for MTEB Benchmark Run", ignore_remote_overrides=True
        )
        local_path = clearml_model.get_local_copy()

        model = AutoLednikModel.from_pretrained(
            local_path, weights_prefix="student.", strict_prefix=True
        )

        tokenizer, _ = load_tokenizer_from_clearml(tokenizer_id=tokenizer_id, task=task)

        model_wrapper = MTEBModelWrapper(
            model=model,
            tokenizer=tokenizer,
            max_length=max_sequence_length,
            model_id=model_id,
            model_name_for_meta=clearml_model.name,
        )
    else:
        clearml_model = None
        model_wrapper = mteb.get_model(model_id)

    _setup_output_file(output_file)

    tasks2run = MTEBTasks(
        [
            task
            for task in mteb.get_tasks(
                languages=["rus"],
                exclusive_language_filter=True,
            )
            if len(task.languages) == 1
            and task.modalities == ["text"]
            and task.metadata.name not in too_big_bench
        ]  # ty:ignore[invalid-argument-type]
    )
    run_result = mteb.evaluate(
        model_wrapper,
        tasks2run,
        encode_kwargs={"batch_size": batch_size},
    )
    run_scores = _calculate_scores(run_result)

    _log_result(
        avg_score=run_scores.avg_score,
        avg_score_task=run_scores.avg_score_task,
        avg_score_per_task=run_scores.avg_score_per_task,
        task=task,
        clearml_model=clearml_model,
    )
    _dump_result(
        avg_score_per_task=run_scores.avg_score_per_task,
        avg_score=run_scores.avg_score,
        avg_score_task=run_scores.avg_score_task,
        model_name=run_result.model_name,
        model_revision=run_result.model_revision,
        output_file=output_file,
        task_id=task.task_id,
        num_parameters=model_wrapper.mteb_model_meta.n_parameters,
    )
    return


@click.command(context_settings={"show_default": True})
@click.option(
    "--model-id",
    type=str,
    required=True,
    help=(
        "Model identifier: either a HuggingFace model id (when --no-clearml) "
        "or a ClearML Model ID (when --clearml)."
    ),
)
@click.option(
    "--clearml/--no-clearml",
    "is_clearml_model",
    default=False,
    help="Treat --model-id as a ClearML model_id and load via ClearML.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to a .jsonl file to append results to (one line per run).",
)
@click.option(
    "--tokenizer-id",
    type=str,
    default=None,
    help=(
        "ClearML Model ID for the tokenizer. Required when --clearml; "
        "ignored when --no-clearml."
    ),
)
@click.option(
    "--max-seq-len",
    "max_sequence_length",
    type=int,
    default=None,
    help=(
        "Maximum sequence length (tokens) for encoding. "
        "If not set, the wrapper/model defaults are used."
    ),
)
@click.option(
    "--batch-size",
    type=int,
    default=128,
    show_default=True,
    help="Batch size for MTEB encode() (encode_kwargs.batch_size).",
)
def cli(
    model_id: str,
    is_clearml_model: bool,
    output_file: Path,
    tokenizer_id: str | None,
    max_sequence_length: int | None,
    batch_size: int,
) -> None:
    """Run the MTEB benchmark on Russian text-only tasks."""
    main(
        model_id=model_id,
        is_clearml_model=is_clearml_model,
        output_file=output_file,
        tokenizer_id=tokenizer_id,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
    )
    return


if __name__ == "__main__":
    cli()
