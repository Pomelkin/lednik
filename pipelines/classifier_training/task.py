from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Literal

import click
import lightning as L
import torch
from clearml import Task
from kostyl.ml.clearml.pulling_utils import get_model_from_clearml
from kostyl.ml.clearml.pulling_utils import get_tokenizer_from_clearml
from kostyl.ml.configs import DDPStrategyConfig
from kostyl.ml.configs import FSDP1StrategyConfig
from kostyl.ml.configs import SingleDeviceStrategyConfig
from kostyl.ml.lightning.callbacks import ClearMLRegistryUploaderCallback
from kostyl.ml.lightning.callbacks import setup_checkpoint_callback
from kostyl.ml.lightning.callbacks import setup_early_stopping_callback
from kostyl.ml.lightning.loggers.tb_logger import setup_tb_logger
from kostyl.utils.logging import setup_logger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.nn import Module

from lednik.distill.training.training_modules import ClassifierTrainingModule
from lednik.static_embeddings import StaticEmbeddingsForSequenceClassification
from pipelines.classifier_training.configs import TrainConfig
from pipelines.classifier_training.configs import TrainingSettings
from pipelines.classifier_training.datamodule import DataModule


torch.set_float32_matmul_precision("high")

logger = setup_logger(fmt="only_message")


def _parse_fast_dev_run(fast_dev_run: str) -> int:
    if fast_dev_run == "":
        return 0
    try:
        val = int(fast_dev_run)
        if val < 0:
            raise ValueError()
        return val
    except ValueError:
        raise click.BadParameter(
            f"Invalid value for --fast-dev-run: {fast_dev_run}. It must be a non-negative integer."
        ) from None


def _validate_input(
    remote_execution_queue: str, fast_dev_run: int, profile: bool
) -> None:
    if remote_execution_queue != "" and fast_dev_run > 0:
        raise ValueError(
            "Cannot use `fast-dev-run` with remote execution. Please set `fast-dev-run` to 0 or disable remote execution."
        )

    if profile and remote_execution_queue != "":
        raise ValueError(
            "Cannot use profiling with remote execution. Please disable profiling."
        )

    if profile and fast_dev_run == 0:
        raise ValueError(
            "Cannot use profiling without `fast-dev-run`. Please set `fast-dev-run` to a positive value."
        )
    return


@dataclass
class _Callbacks:
    checkpoint: ModelCheckpoint
    lr_monitor: LearningRateMonitor
    model_uploader: ClearMLRegistryUploaderCallback
    early_stopping: EarlyStopping | None = None

    def to_list(self) -> list[Callback]:
        """Convert dataclass fields to a list of Callbacks. None values are omitted."""
        return [
            getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]


def _setup_callbacks(
    task: Task,
    root_path: Path,
    training_settings: TrainingSettings,
    output_model_name: str,
    output_model_tags: list[str] | None = None,
) -> _Callbacks:
    lr_monitor = LearningRateMonitor(
        logging_interval="step", log_weight_decay=True, log_momentum=True
    )
    checkpoint_callback = setup_checkpoint_callback(
        root_path / "checkpoints" / task.name / task.id,
        training_settings.checkpoint,
    )
    model_uploader = ClearMLRegistryUploaderCallback(
        task=task,
        output_model_name=output_model_name,
        output_model_tags=output_model_tags,
        ckpt_callback=checkpoint_callback,
        verbose=True,
        uploading_frequency="after-every-eval",
    )
    if training_settings.early_stopping is not None:
        early_stopping_callback = setup_early_stopping_callback(
            training_settings.early_stopping
        )
    else:
        early_stopping_callback = None

    callbacks = _Callbacks(
        checkpoint=checkpoint_callback,
        lr_monitor=lr_monitor,
        model_uploader=model_uploader,
        early_stopping=early_stopping_callback,
    )
    return callbacks


def _setup_loggers(task: Task, root_path: Path) -> list[TensorBoardLogger]:
    loggers = [
        setup_tb_logger(root_path / "runs" / task.name / task.id),
    ]
    return loggers


def _setup_strategy(
    training_settings: TrainingSettings,
    auto_wrap_policy: set[type[Module]] | None = None,
) -> Literal["auto"] | FSDPStrategy | DDPStrategy:
    if isinstance(training_settings.trainer.devices, list):
        num_devices = len(training_settings.trainer.devices)
    else:
        num_devices = training_settings.trainer.devices

    match training_settings.trainer.strategy:
        case FSDP1StrategyConfig():
            if num_devices < 2:
                raise ValueError("FSDP strategy requires multiple devices.")

            if auto_wrap_policy is None:
                raise ValueError("auto_wrap_policy must be provided for FSDP strategy.")

            mixed_precision_config = MixedPrecision(
                param_dtype=getattr(
                    torch, training_settings.trainer.strategy.param_dtype
                ),
                reduce_dtype=getattr(
                    torch, training_settings.trainer.strategy.reduce_dtype
                ),
                buffer_dtype=getattr(
                    torch, training_settings.trainer.strategy.buffer_dtype
                ),
            )
            strategy = FSDPStrategy(
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_config,
            )
        case DDPStrategyConfig():
            if num_devices < 2:
                raise ValueError("DDP strategy requires at least two devices.")
            strategy = DDPStrategy(
                find_unused_parameters=training_settings.trainer.strategy.find_unused_parameters
            )
        case SingleDeviceStrategyConfig():
            if num_devices != 1:
                raise ValueError("SingleDevice strategy requires exactly one device.")
            strategy = "auto"
        case _:
            raise ValueError(
                f"Unsupported strategy type: {type(training_settings.trainer.strategy)}"
            )
    return strategy


def _parse_tags(ctx: click.Context, param: click.Parameter, value: str) -> list[str]:
    """Parse comma-separated tags into a list."""
    if not value:
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


@click.command()
@click.option(
    "--remote-execution-queue",
    type=click.STRING,
    default="",
    help="Queue for remotely executing task on ClearML. If empty, the training will be run locally.",
)
@click.option(
    "--fast-dev-run",
    type=click.STRING,
    default="0",
    help="Run only a few batches for quick testing. If set to 0 - fast-dev-run will be disabled.",
)
@click.option(
    "--profile",
    is_flag=True,
    help="Enable profiling for the training run.",
    default=False,
)
@click.option(
    "--tags",
    type=click.STRING,
    default="",
    callback=_parse_tags,
    help="Additional tags for the task, separated by commas (e.g., 'tag1,tag2,tag3').",
)
def _finetune_static_model(
    remote_execution_queue: str,
    fast_dev_run: str,
    profile: bool,
    tags: list[str],
) -> None:
    fast_dev_run_ = _parse_fast_dev_run(fast_dev_run)
    _validate_input(remote_execution_queue, fast_dev_run_, profile)
    if fast_dev_run_ == 0:
        fast_dev_run_ = False

    task: Task = Task.init(
        project_name="Lednik",
        task_name="Classifier Training (Static Embeddings)",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=True,
        auto_connect_frameworks={
            "pytorch": False,
            "tensorboard": True,
            "matplotlib": True,
            "detect_repository": True,
        },
        tags=["classification", *tags],
    )

    ROOT_PATH = Path(__file__).parent.parent.parent

    train_config = TrainConfig.connect_as_dict(
        task, ROOT_PATH / "configs" / "finetuning" / "train_config.yaml"
    )
    train_settings = TrainingSettings.connect_as_file(
        task, ROOT_PATH / "configs" / "finetuning" / "training_settings.yaml"
    )

    if remote_execution_queue != "":
        task.execute_remotely(queue_name=remote_execution_queue, exit_process=True)

    if profile:
        from lightning.pytorch.profilers import SimpleProfiler

        profiler = SimpleProfiler(
            filename=f"{task.name}_{task.id}",
        )
    else:
        profiler = None

    classifier, clearml_model = get_model_from_clearml(
        model_id=train_settings.model_id,
        model=StaticEmbeddingsForSequenceClassification,
        name="Static Embeddings (Initial model)",
        task=task,
        embedding_dropout=train_config.embedding_dropout,
        classifier_dropout=train_config.classifier_dropout,
        id2label=train_config.id2label,
        label2id=train_config.label2id,
        num_labels=train_config.num_labels,
    )

    tokenizer = classifier.model.tokenizer
    if tokenizer is None:
        if train_settings.tokenizer_id is None:
            raise ValueError(
                "Tokenizer is not found in the model. Please provide a tokenizer_id in the training settings."
            )
        tokenizer, _ = get_tokenizer_from_clearml(
            model_id=train_settings.tokenizer_id,
            task=task,
            name=f"{clearml_model.name} Tokenizer",
        )
    datamodule = DataModule(data_cfg=train_settings.data, tokenizer=tokenizer)

    training_module = ClassifierTrainingModule(
        model=classifier,
        config=train_config,
        task=task,
    )

    callbacks = _setup_callbacks(
        task,
        ROOT_PATH,
        training_settings=train_settings,
        output_model_name=clearml_model.name + " For Seq Classification",
        output_model_tags=["Classifier", "Static Embeddings"],
    )
    loggers = _setup_loggers(task, ROOT_PATH)
    strategy = _setup_strategy(training_settings=train_settings)

    trainer = L.Trainer(
        max_epochs=train_settings.trainer.max_epochs,
        accelerator=train_settings.trainer.accelerator,
        devices=train_settings.trainer.devices,
        strategy=strategy,
        precision=train_settings.trainer.precision,
        accumulate_grad_batches=train_settings.trainer.accumulate_grad_batches,
        gradient_clip_val=None,
        val_check_interval=train_settings.trainer.val_check_interval,
        callbacks=callbacks.to_list(),
        log_every_n_steps=train_settings.trainer.log_every_n_steps,
        limit_train_batches=train_settings.trainer.limit_train_batches,
        limit_val_batches=train_settings.trainer.limit_val_batches,
        limit_test_batches=train_settings.trainer.limit_test_batches,
        limit_predict_batches=train_settings.trainer.limit_predict_batches,
        logger=loggers,
        fast_dev_run=fast_dev_run_,
        profiler=profiler,
    )

    trainer.fit(training_module, datamodule=datamodule)
    return


if __name__ == "__main__":
    _finetune_static_model()
