from pathlib import Path
from typing import cast

import click
import lightning as L
import torch
from clearml import Task
from kostyl.ml.integrations.clearml import load_model_from_clearml
from kostyl.ml.integrations.clearml import load_tokenizer_from_clearml
from kostyl.utils.logging import setup_logger
from transformers import AutoModel
from transformers import PreTrainedModel

from lednik.distill.training_module import DistillationModule
from lednik.models import MODEL_MAPPING
from pipelines.distill.configs import DistillationConfig
from pipelines.distill.configs import TrainingSettings
from pipelines.distill.datamodule import DataModule
from pipelines.utils import CheckpointUploaderConfig
from pipelines.utils import setup_callbacks
from pipelines.utils import setup_loggers
from pipelines.utils import setup_strategy


torch.set_float32_matmul_precision("high")

logger = setup_logger(fmt="only_message")


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
    "--tags",
    type=click.STRING,
    default="",
    callback=_parse_tags,
    help="Additional tags for the task, separated by commas (e.g., 'tag1,tag2,tag3').",
)
def _distill_model(
    remote_execution_queue: str,
    tags: list[str],
) -> None:
    task: Task = Task.init(
        project_name="Lednik",
        task_name="Model Distillation",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=True,
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
    teacher, clearml_teacher = load_model_from_clearml(
        model_id=training_settings.teacher_model_id,
        model=AutoModel,  # pyright: ignore[reportArgumentType]
        name="Teacher Model",
        task=task,
    )
    teacher = cast(PreTrainedModel, teacher)

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
        model_id=training_settings.tokenizer_id,
        task=task,
        name="Tokenizer",
    )

    ### Distillation Module Setup ###
    distillation_module = DistillationModule(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        train_cfg=distill_config,
        strategy_config=training_settings.trainer.strategy,
        task=task,
        num_labels=training_settings.data.val_num_labels,
        redis_config=training_settings.redis,
    )

    ### Callbacks, Loggers, and Strategy Setup ###
    ckpt_uploader_config = CheckpointUploaderConfig(
        model_name=clearml_student.name,
        config_dict=student.config.to_diff_dict(),
        upload_as_new_model=False,
        tags=[*clearml_student.tags],
        framework="PyTorch",
        comment=f"Model distilled from {clearml_teacher.id}.",
    )
    callbacks = setup_callbacks(
        task=task,
        root_path=ROOT_PATH,
        checkpoint_cfg=training_settings.checkpoint,
        early_stopping_cfg=training_settings.early_stopping,
        checkpoint_uploader_config=ckpt_uploader_config,
    )
    loggers = setup_loggers(task, ROOT_PATH)
    strategy = setup_strategy(
        strategy_settings=training_settings.trainer.strategy,
        devices=training_settings.trainer.devices,
    )

    datamodule = DataModule(config=training_settings.data, tokenizer=tokenizer)

    trainer = L.Trainer(
        max_epochs=training_settings.trainer.max_epochs,
        accelerator=training_settings.trainer.accelerator,
        devices=training_settings.trainer.devices,
        strategy=strategy,
        precision=training_settings.trainer.precision,
        accumulate_grad_batches=training_settings.trainer.accumulate_grad_batches,
        gradient_clip_val=None,
        val_check_interval=training_settings.trainer.val_check_interval,
        callbacks=callbacks.to_list(),
        log_every_n_steps=training_settings.trainer.log_every_n_steps,
        limit_train_batches=training_settings.trainer.limit_train_batches,
        limit_val_batches=training_settings.trainer.limit_val_batches,
        limit_test_batches=training_settings.trainer.limit_test_batches,
        limit_predict_batches=training_settings.trainer.limit_predict_batches,
        logger=loggers,
    )

    trainer.fit(distillation_module, datamodule=datamodule)
    return


if __name__ == "__main__":
    _distill_model()
