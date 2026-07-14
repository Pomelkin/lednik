from pathlib import Path

import click
import lightning as L
import torch
from clearml import InputModel
from clearml import OutputModel
from clearml import Task
from kostyl.ml.configs import SUPPORTED_STRATEGIES
from kostyl.ml.configs import DDPStrategyConfig
from kostyl.ml.configs import FSDP1StrategyConfig
from kostyl.ml.configs import SingleDeviceStrategyConfig
from kostyl.ml.configs.structs.training_settings import FSDP2StrategyConfig
from kostyl.ml.integrations.clearml import load_model_from_clearml
from kostyl.ml.integrations.clearml import load_tokenizer_from_clearml
from kostyl.ml.integrations.lightning.callbacks import setup_checkpoint_callback
from kostyl.ml.integrations.lightning.callbacks import setup_early_stopping_callback
from kostyl.ml.integrations.lightning.loggers import ClearMLLogger
from kostyl.utils.logging import setup_logger
from lightning import Callback
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.accelerators import CPUAccelerator
from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.pytorch.accelerators import MPSAccelerator
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning.pytorch.strategies import Strategy
from transformers import TokenizersBackend

from lednik.distill.training_module import DistillationModule
from lednik.models import MODEL_MAPPING
from pipelines.distill.configs import DistillationConfig
from pipelines.distill.configs import TrainingSettings
from pipelines.distill.datamodule import DataModule


torch.set_float32_matmul_precision("high")

logger = setup_logger(fmt="only_message")


def _parse_tags(ctx: click.Context, param: click.Parameter, value: str) -> list[str]:
    """Parse comma-separated tags into a list."""
    if not value:
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def _choose_accelerator(accelerator: str) -> type[Accelerator]:
    """Choose the appropriate PyTorch Lightning Accelerator based on the input string."""
    accelerator = accelerator.lower()
    match accelerator:
        case "cuda":
            accelerator_class = CUDAAccelerator
        case "mps":
            accelerator_class = MPSAccelerator
        case "cpu":
            accelerator_class = CPUAccelerator
        case _:
            raise ValueError(f"Unsupported accelerator: {accelerator}")
    return accelerator_class


def setup_strategy(  # noqa: C901
    accelerator: str,
    strategy_settings: SUPPORTED_STRATEGIES,
    devices: list[int] | int,
) -> Strategy:
    """Configure and return a PyTorch Lightning training strategy."""
    if isinstance(devices, list):
        if len(devices) == 0:
            raise ValueError("Device list cannot be empty.")
        num_devices = len(devices)
        device_ids = devices
    else:
        num_devices = devices
        device_ids = list(range(num_devices))

    accelerator_class = _choose_accelerator(accelerator)
    if not accelerator_class.is_available():
        raise ValueError(f"{accelerator_class.name()} accelerator is not available.")

    parallel_devices: list[torch.device] = accelerator_class.get_parallel_devices(
        device_ids
    )

    match strategy_settings:
        case DDPStrategyConfig():
            if num_devices < 2:
                raise ValueError("DDP strategy requires at least two devices.")
            strategy = DDPStrategy(
                accelerator=accelerator,
                parallel_devices=parallel_devices,
                find_unused_parameters=strategy_settings.find_unused_parameters,
            )
        case FSDP1StrategyConfig():
            if num_devices < 2:
                raise ValueError("FSDP strategy requires multiple devices.")
            strategy = FSDPStrategy(
                accelerator=accelerator,  # Accelerator is already handled by the Trainer
                parallel_devices=parallel_devices,
            )
        case FSDP2StrategyConfig():
            if num_devices < 2:
                raise ValueError("FSDP strategy requires multiple devices.")
            strategy = ModelParallelStrategy(
                data_parallel_size=num_devices,
                tensor_parallel_size=1,
                save_distributed_checkpoint=False,
            )
            strategy.parallel_devices = parallel_devices

        case SingleDeviceStrategyConfig():
            if num_devices != 1:
                raise ValueError("SingleDevice strategy requires exactly one device.")

            strategy = SingleDeviceStrategy(
                device=parallel_devices[0], accelerator=accelerator
            )
        case _:
            raise ValueError(
                f"Unsupported strategy: {strategy_settings.__class__.__name__}"
            )
    return strategy


@click.command()
@click.option(
    "--remote-execution-queue",
    type=click.STRING,
    default="",
    help="Queue for remotely executing task on ClearML. If empty, the training will be run locally.",
)
@click.option(
    "--tags",
    type=click.STRING,
    default="",
    callback=_parse_tags,
    help="Additional tags for the task, separated by commas (e.g., 'tag1,tag2,tag3').",
)
@click.option(
    "--reuse-last-task-id/--no-reuse-last-task-id",
    "reuse_last_task_id",
    default=True,
    help="Whether to reuse the last task ID for this task. Useful for resuming or updating an existing task in ClearML.",
)
def _distill_model(
    remote_execution_queue: str,
    tags: list[str],
    reuse_last_task_id: bool = True,
) -> None:
    task: Task = Task.init(
        project_name="Lednik",
        task_name="Model Distillation",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=reuse_last_task_id,
        auto_connect_frameworks={
            "pytorch": False,
            "tensorboard": True,
            "matplotlib": True,
            "detect_repository": True,
        },
        tags=tags,
    )

    ROOT_PATH = Path(__file__).parent.parent.parent

    distill_config = DistillationConfig.connect_as_file(
        task, ROOT_PATH / "configs" / "distill_config.yaml"
    )
    training_settings = TrainingSettings.connect_as_file(
        task, ROOT_PATH / "configs" / "training_settings.yaml"
    )

    if remote_execution_queue != "":
        task.execute_remotely(queue_name=remote_execution_queue, exit_process=True)

    ### Teacher Model Loading ###
    clearml_teacher = InputModel(model_id=training_settings.teacher_model_id)

    ### Student Model Loading ###
    model_cls = MODEL_MAPPING[training_settings.model_cfg.model_type]

    load_kwargs = training_settings.model_cfg.model_dump(exclude={"model_type"})
    if training_settings.is_student_lightning_checkpoint:
        load_kwargs.update(
            {"weights_prefix": training_settings.checkpoint_weight_prefix}
        )

    student, clearml_student = load_model_from_clearml(
        model_id=training_settings.student_model_id,
        model=model_cls,
        task=task,
        name="Model to Distill (Student)",
        **load_kwargs,
    )

    ### Tokenizer Loading ###
    tokenizer, _ = load_tokenizer_from_clearml(
        tokenizer_id=training_settings.tokenizer_id,
        task=task,
        name="Tokenizer",
    )
    if not isinstance(tokenizer, (TokenizersBackend)):
        raise TypeError(
            f"Loaded tokenizer is not an instance of TokenizersBackend or SentencePieceBackend. Got {type(tokenizer)}"
        )

    ### Data and distill modules setup ###
    distillation_module = DistillationModule(
        teacher_hidden_size=clearml_teacher.config_dict["hidden_size"],
        student=student,
        tokenizer=tokenizer,
        train_cfg=distill_config,
        strategy_config=training_settings.trainer.strategy,
        task=task,
        num_labels=training_settings.data.val_num_labels,
        redis_config=training_settings.redis,
        runner_config=training_settings.runner_config,
    )

    datamodule = DataModule(config=training_settings.data, tokenizer=tokenizer)

    ### Callbacks, Loggers, and Strategy ###
    output_model = OutputModel(
        task=task,
        name=clearml_student.name,
        tags=[tag for tag in clearml_student.tags if tag != "Not Distilled"],
        framework="PyTorch",
        comment=f"Model distilled from {clearml_teacher.id}.",
    )
    logger = ClearMLLogger(
        task=task,
        output_model=output_model,
        upload_checkpoints=False,
        upload_strategy="best",
        model_config_provider=lambda: distillation_module.model_config,
    )

    lr_monitor = LearningRateMonitor(
        logging_interval="step", log_weight_decay=True, log_momentum=False
    )
    checkpoint_callback = setup_checkpoint_callback(
        ROOT_PATH / "checkpoints" / task.name / task.id,
        training_settings.checkpoint,
    )
    callbacks: list[Callback] = [checkpoint_callback, lr_monitor]
    if training_settings.early_stopping is not None:
        early_stopping_callback = setup_early_stopping_callback(
            training_settings.early_stopping
        )
        callbacks.append(early_stopping_callback)

    strategy = setup_strategy(
        accelerator=training_settings.trainer.accelerator,
        strategy_settings=training_settings.trainer.strategy,
        devices=training_settings.trainer.devices,
    )

    ### Trainer Setup and Training Start ###
    trainer = L.Trainer(
        max_epochs=training_settings.trainer.max_epochs,
        accelerator=training_settings.trainer.accelerator
        if strategy.accelerator is None
        else "auto",
        devices=training_settings.trainer.devices,
        strategy=strategy,
        precision=training_settings.trainer.precision,
        accumulate_grad_batches=training_settings.trainer.accumulate_grad_batches,
        gradient_clip_val=None,
        val_check_interval=training_settings.trainer.val_check_interval,
        callbacks=callbacks,
        log_every_n_steps=training_settings.trainer.log_every_n_steps,
        limit_train_batches=training_settings.trainer.limit_train_batches,
        limit_val_batches=training_settings.trainer.limit_val_batches,
        limit_test_batches=training_settings.trainer.limit_test_batches,
        limit_predict_batches=training_settings.trainer.limit_predict_batches,
        logger=[logger],
    )

    trainer.fit(distillation_module, datamodule=datamodule)
    return


if __name__ == "__main__":
    _distill_model()
