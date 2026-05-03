from typing import Literal
from collections import defaultdict
from typing import Any
from mteb.get_tasks import MTEBTasks
from pathlib import Path
import click
import mteb
from kostyl.ml.integrations.clearml import (
    load_model_from_clearml,
    load_tokenizer_from_clearml,
)
from lednik.models import (
    LednikModel,
    StaticEmbeddingsModel,
    LednikConfig,
    StaticEmbeddingsConfig,
)
from bench.mteb import MTEBModelWrapper
from clearml import InputModel, Task
from kostyl.utils import setup_logger
from transformers import AutoModel
from mteb.results.model_result import ModelResult


logger = setup_logger()

model_mapping = {
    StaticEmbeddingsConfig.model_type: StaticEmbeddingsModel,
    LednikConfig.model_type: LednikModel,
}

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
}


def _determine_lednik_model_type(
    model_id: str,
) -> type[LednikModel] | type[StaticEmbeddingsModel] | type[AutoModel]:
    clearml_model = InputModel(model_id=model_id)
    model_type = clearml_model.config_dict.get("model_type", None)
    if model_type is None:
        raise ValueError(
            f"Model type not found in ClearML model config for model_id={model_id}. "
            "Make sure the model was logged with the correct configuration."
        )
    model_class = model_mapping.get(model_type, AutoModel)
    logger.info(
        f"Determined model class for model_id={model_id[:10]}: {model_class.__name__}"
    )
    return model_class


def _setup_output_file(path: Path) -> None:
    if path.suffix != ".jsonl":
        raise ValueError("Output file must have .jsonl extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    return


def _dump_result(
    result: ModelResult, output_file: Path, task_id: str, num_parameters: int | None
) -> None:
    data2log: dict[str, Any] = {
        "TaskID": task_id,
        "ModelName": result.model_name,
        "ModelRevision": result.model_revision,
        "NumParameters": f"{num_parameters / 1_000_000:.2f}M"
        if num_parameters is not None
        else None,
    }
    task_type2metric: dict[str, dict[str, float]] = defaultdict(dict)
    score_sum = 0.0
    task_type2sum: dict[str, float] = defaultdict(float)
    for task_result in result.task_results:
        task_type2metric[task_result.task_type][task_result.task_name] = (
            task_result.main_score
        )
        score_sum += task_result.main_score
        task_type2sum[task_result.task_type] += task_result.main_score

    avg_per_task = {
        f"Avg{task_type}": task_type2sum[task_type] / len(metrics)
        for task_type, metrics in task_type2metric.items()
    }
    avg_per_task.update(
        {
            "AvgScore": score_sum / len(result.task_results)
            if result.task_results
            else 0.0,
        }
    )
    data2log.update(avg_per_task)
    data2log.update(
        {
            f"{task_type}/{task_name}": score
            for task_type, metrics in task_type2metric.items()
            for task_name, score in metrics.items()
        }
    )

    with output_file.open("a", encoding="utf-8") as f:
        f.write(f"{data2log}\n")
    return


def _log_result(
    task: Task, result: ModelResult, clearml_model: InputModel | None = None
) -> None:
    task_type2metric: dict[str, dict[str, float]] = defaultdict(dict)
    score_sum = 0.0
    task_type2sum: dict[str, float] = defaultdict(float)
    for task_result in result.task_results:
        task_type2metric[task_result.task_type][task_result.task_name] = (
            task_result.main_score
        )
        score_sum += task_result.main_score
        task_type2sum[task_result.task_type] += task_result.main_score

    avg_per_task = {
        f"Avg{task_type}": task_type2sum[task_type] / len(metrics)
        for task_type, metrics in task_type2metric.items()
    }
    avg_per_task.update(
        {
            "AvgScore": score_sum / len(result.task_results)
            if result.task_results
            else 0.0,
        }
    )

    task_logger = task.get_logger()
    for task_type, avg in avg_per_task.items():
        task_logger.report_single_value(name=task_type, value=avg)

    for task_type, metrics in task_type2metric.items():
        for task_name, score in metrics.items():
            task_logger.report_single_value(
                name=f"{task_type}/{task_name}", value=score
            )

    if clearml_model is not None and not clearml_model.published:
        for task_type, avg in avg_per_task.items():
            clearml_model.report_single_value(name=task_type, value=avg)

    logger.info(f"Avg Score: {avg_per_task['AvgScore']:.4f}")
    return


def main(
    model_id: str,
    is_clearml_model: bool,
    output_file: Path,
    tokenizer_id: str | None = None,
    max_sequence_length: int | None = None,
    batch_size: int = 128,
    pooling: Literal["mean", "cls", "last"] = "mean",
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
        model_cls = _determine_lednik_model_type(model_id)

        kwargs = (
            {"weights_prefix": "student"}
            if "LightningCheckpoint" in InputModel(model_id).tags
            else {}
        )  # TODO: remove this kostyl
        model, clearml_model = load_model_from_clearml(
            model_id=model_id,
            model=model_cls,
            task=task,
            **kwargs,  # type: ignore
        )

        tokenizer, _ = load_tokenizer_from_clearml(model_id=tokenizer_id, task=task)

        model_wrapper = MTEBModelWrapper(
            model=model,
            tokenizer=tokenizer,
            max_length=max_sequence_length,
            model_id=model_id,
            pooling=pooling,
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
            and task.__class__.__name__ not in too_big_bench
        ]
    )

    run_result = mteb.evaluate(
        model_wrapper,  # type: ignore
        tasks2run,
        encode_kwargs={"batch_size": batch_size},
    )

    _log_result(task, run_result, clearml_model)
    _dump_result(
        run_result,
        output_file,
        task_id=task.task_id,
        num_parameters=model_wrapper.mteb_model_meta.n_parameters
        if getattr(model_wrapper, "mteb_model_meta", None)
        else None,
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
@click.option(
    "--pooling",
    type=click.Choice(["mean", "cls", "last"], case_sensitive=False),
    default="mean",
    show_default=True,
    help="Pooling strategy to use for the model wrapper when handling token embeddings.",
)
def cli(
    model_id: str,
    is_clearml_model: bool,
    output_file: Path,
    tokenizer_id: str | None,
    max_sequence_length: int | None,
    batch_size: int,
    pooling: Literal["mean", "cls", "last"],
) -> None:
    """Run the MTEB benchmark on Russian text-only tasks."""
    main(
        model_id=model_id,
        is_clearml_model=is_clearml_model,
        output_file=output_file,
        tokenizer_id=tokenizer_id,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        pooling=pooling,
    )
    return


if __name__ == "__main__":
    cli()
