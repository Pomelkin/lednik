from pathlib import Path
from typing import Literal
from typing import override

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from kostyl.ml.integrations.clearml import collect_clearml_datasets
from kostyl.ml.integrations.clearml import download_clearml_datasets
from kostyl.ml.integrations.clearml import get_datasets_paths
from kostyl.utils.logging import setup_logger
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from sage.spelling_corruption import SBSCConfig
from sage.spelling_corruption import SBSCCorruptor
from torch.utils.data import DataLoader
from transformers import TokenizersBackend

from lednik.distill.collator import ContrastiveCollator

from .configs import DataConfig


logger = setup_logger(fmt="only_message")


class DataModule(LightningDataModule):
    """A DataModule standardizes the training, val, test splits, data preparation and transforms."""

    def __init__(
        self,
        config: DataConfig | dict,
        tokenizer: TokenizersBackend,
    ) -> None:
        """
        Initializes the DataModule for fine-tuning.

        This constructor sets up the configuration, tokenizer, and dataset references required for
        training and validation. It handles the resolution of ClearML dataset references and
        prepares the internal state for dataset loading.

        Args:
            config (DataConfig | dict): The configuration object or dictionary containing
                dataset parameters, batch sizes, and column mappings. If a dictionary is provided,
                it is validated and converted to a `DataConfig` object.
            tokenizer (TokenizersBackend | SentencePieceBackend): The tokenizer instance used for processing
                text data.

        """
        super().__init__()
        if isinstance(config, dict):
            config = DataConfig.model_validate(config)
        self.save_hyperparameters({"data_cfg": config.model_dump()})

        self.clearml_datasets = collect_clearml_datasets(config.datasets)

        self.train_collator = ContrastiveCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            query_tok_colname=config.query_tok_colname,
            query_text_colname=config.query_text_colname,
            pos_tok_colname=config.pos_tok_colname,
            pos_text_colname=config.pos_text_colname,
            neg_tok_colname=config.neg_tok_colname,
            neg_text_colname=config.neg_text_colname,
            query_teacher_embedding_colname=config.query_teacher_embedding_colname,
            pos_teacher_embedding_colname=config.pos_teacher_embedding_colname,
            neg_teacher_embedding_colname=config.neg_teacher_embedding_colname,
            label_colname=config.val_label_colname,
            aug_prob=config.aug_prob,
            corruptor=SBSCCorruptor.from_config(SBSCConfig(min_typos=5))
            if config.aug_prob > 0.0
            else None,
        )
        self.val_collator = ContrastiveCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            query_tok_colname=config.query_tok_colname,
            query_text_colname=config.query_text_colname,
            pos_tok_colname=config.pos_tok_colname,
            pos_text_colname=config.pos_text_colname,
            neg_tok_colname=config.neg_tok_colname,
            neg_text_colname=config.neg_text_colname,
            query_teacher_embedding_colname=config.query_teacher_embedding_colname,
            pos_teacher_embedding_colname=config.pos_teacher_embedding_colname,
            neg_teacher_embedding_colname=config.neg_teacher_embedding_colname,
            label_colname=config.val_label_colname,
            aug_prob=0.0,  # No augmentation during validation
            corruptor=None,
        )

        self.config = config
        self.path_to_ds_name: dict[Path, str] = {}
        self.datasets_paths: dict[str, Path] | None = None
        self.train_dataset: HFDataset | None = None
        self.val_dataset: HFDataset | None = None
        self.test_dataset: HFDataset | None = None
        return

    @override
    def prepare_data(self) -> None:
        download_clearml_datasets(list(self.clearml_datasets.values()))
        return

    def _collect_subset_paths(
        self,
        datasets_paths: dict[str, Path],
        split_name: str,
    ) -> list[Path]:
        split_paths: list[Path] = []
        for dataset_name, dataset_path in datasets_paths.items():
            found_paths = list(dataset_path.glob(f"**/{split_name}/"))
            if len(found_paths) == 0:
                logger.warning(
                    f"No {split_name} data found for dataset {dataset_name} at path {dataset_path}."
                )
                continue

            for path in found_paths:
                self.path_to_ds_name[path] = dataset_name
                split_paths.append(path)

        if len(split_paths) == 0:
            raise ValueError(
                f"No {split_name} data found in any of the provided datasets: {'\n -'.join(list(datasets_paths.keys()))}"
            )
        return split_paths

    def _concat_splits(
        self, subset_paths: list[Path], data_columns: set[str] | None = None
    ) -> HFDataset:
        datasets: list[HFDataset] = []
        for subset_path in subset_paths:
            ds = HFDataset.load_from_disk(subset_path)
            if data_columns is not None:
                col2remove = [col for col in ds.column_names if col not in data_columns]
            else:
                col2remove = []

            if len(col2remove) == len(ds.column_names):
                logger.warning(
                    f"None of the specified data_columns {data_columns} belong "
                    f"to the dataset {self.path_to_ds_name.get(subset_path, 'Unknown')}."
                )
                continue
            else:
                ds = ds.remove_columns(col2remove)
                datasets.append(ds)
        if len(datasets) == 0:
            raise ValueError(
                f"Not one of the datasets: {', '.join(set(self.path_to_ds_name.values()))}, "
                f"doesn't have the specified data_columns {data_columns}"
            )
        return concatenate_datasets(datasets)

    def _create_hf_dataset(self, subset: Literal["train", "val"]) -> HFDataset:
        if self.datasets_paths is None:
            self.datasets_paths = get_datasets_paths(self.clearml_datasets)
        data_columns = {
            self.config.query_tok_colname,
            self.config.pos_tok_colname,
            self.config.query_teacher_embedding_colname,
            self.config.pos_teacher_embedding_colname,
        }
        if (
            self.config.neg_tok_colname is not None
            and self.config.neg_teacher_embedding_colname is not None
        ):
            data_columns.add(self.config.neg_tok_colname)
            data_columns.add(self.config.neg_teacher_embedding_colname)
        if subset == "val" and self.config.val_label_colname is not None:
            data_columns.add(self.config.val_label_colname)

        subset_paths = self._collect_subset_paths(self.datasets_paths, subset)
        dataset = self._concat_splits(subset_paths, data_columns=data_columns)
        return dataset

    @override
    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                if self.train_dataset is None:
                    self.train_dataset = self._create_hf_dataset(subset="train")
                if self.val_dataset is None:
                    self.val_dataset = self._create_hf_dataset(subset="val")
            case "validate":
                if self.val_dataset is None:
                    self.val_dataset = self._create_hf_dataset(subset="val")
            case _:
                raise ValueError(f"Unsupported stage: {stage}")
        return

    @override
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_dataset is None:
            raise ValueError("Training dataset is not initialized.")
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=self.train_collator,
        )

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized.")
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=self.val_collator,
        )
