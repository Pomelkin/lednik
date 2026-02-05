from pathlib import Path

import click
import lightning as L
import torch
from clearml import Task
from kostyl.ml.integrations.clearml import load_model_from_clearml
from kostyl.ml.integrations.clearml import load_tokenizer_from_clearml
from kostyl.utils.logging import setup_logger

from lednik.distill.training.training_modules import ClassifierTrainingModule
from lednik.models import StaticEmbeddingsForSequenceClassification
from pipelines.classifier_training.configs import ClassifierHyperparamsConfig
from pipelines.classifier_training.configs import TrainingSettings
from pipelines.classifier_training.datamodule import DataModule
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
def _finetune_model(
    remote_execution_queue: str,
    tags: list[str],
) -> None:
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
        tags=tags,
    )

    ROOT_PATH = Path(__file__).parent.parent.parent

    hyperparams_config = ClassifierHyperparamsConfig.connect_as_dict(
        task, ROOT_PATH / "configs" / "classification" / "hyperparams_config.yaml"
    )
    training_settings = TrainingSettings.connect_as_file(
        task,
        ROOT_PATH / "configs" / "classification" / "training_settings.yaml",
        alias="Settings",
    )
    if remote_execution_queue != "":
        task.execute_remotely(queue_name=remote_execution_queue, exit_process=True)

    ### Setup Model  and Training Module ###
    classifier, clearml_model = load_model_from_clearml(
        model_id=training_settings.model_id,
        model=StaticEmbeddingsForSequenceClassification,
        name="Static Embeddings (Initial model)",
        task=task,
        weights_prefix=training_settings.weights_prefix,
        strict_prefix=True,
        embedding_dropout=hyperparams_config.embedding_dropout,
        classifier_dropout=hyperparams_config.classifier_dropout,
        id2label=hyperparams_config.id2label,
        label2id=hyperparams_config.label2id,
        num_labels=hyperparams_config.num_labels,
    )

    training_module = ClassifierTrainingModule(
        model=classifier,
        config=hyperparams_config,
        strategy_config=training_settings.trainer.strategy,
        task=task,
    )

    ### Setup Tokenizer ###
    tokenizer = getattr(classifier.model, "tokenizer", None)
    if tokenizer is None:
        if training_settings.tokenizer_id is None:
            raise ValueError(
                "Tokenizer is not found in the model. Please provide a tokenizer_id in the training settings."
            )
        tokenizer, _ = load_tokenizer_from_clearml(
            model_id=training_settings.tokenizer_id,
            task=task,
            name=f"{clearml_model.name} Tokenizer",
        )

    ### Callbacks, Loggers, and Strategy Setup ###
    ckpt_uploader_config = CheckpointUploaderConfig(
        model_name=clearml_model.name + " ForSeqClassification",
        config_dict=clearml_model.config_dict,
        upload_as_new_model=False,
        framework="PyTorch",
        tags=[*clearml_model.tags, "SeqClassification"],
        label_enumeration=hyperparams_config.label2id,
        comment="Model fine-tuned for sequence classification task.",
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

    datamodule = DataModule(data_cfg=training_settings.data, tokenizer=tokenizer)

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

    trainer.fit(training_module, datamodule=datamodule)
    return


if __name__ == "__main__":
    _finetune_model()
