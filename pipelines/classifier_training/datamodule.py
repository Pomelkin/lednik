from pathlib import Path
from typing import Literal
from typing import override

import lightning as L
from kostyl.ml.data_collator import BatchCollatorWithKeyAlignment
from kostyl.ml.integrations.clearml import collect_clearml_datasets
from kostyl.ml.integrations.clearml import download_clearml_datasets
from kostyl.ml.integrations.clearml import get_datasets_paths
from kostyl.utils.logging import setup_logger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizerBase
from transformers import SentencePieceBackend
from transformers import TokenizersBackend

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from .configs import DataConfig


logger = setup_logger(fmt="only_message")


class DataModule(L.LightningDataModule):
    """A DataModule standardizes the training, val, test splits, data preparation and transforms."""

    def __init__(
        self,
        data_cfg: DataConfig | dict,
        tokenizer: SentencePieceBackend | TokenizersBackend | PreTrainedTokenizerBase,
    ) -> None:
        """
        Initializes the DataModule for fine-tuning.

        This constructor sets up the configuration, tokenizer, and dataset references required for
        training and validation. It handles the resolution of ClearML dataset references and
        prepares the internal state for dataset loading.

        Args:
            data_cfg (DataConfig | dict): The configuration object or dictionary containing
                dataset parameters, batch sizes, and column mappings. If a dictionary is provided,
                it is validated and converted to a `DataConfig` object.
            tokenizer (SentencePieceBackend | TokenizersBackend): The tokenizer instance used for processing
                text data.

        """
        super().__init__()
        if isinstance(data_cfg, dict):
            data_cfg = DataConfig.model_validate(data_cfg)
        self.save_hyperparameters({"data_cfg": data_cfg.model_dump()})

        self.clearml_datasets = collect_clearml_datasets(data_cfg.datasets)

        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.num_workers

        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id)  # type: ignore
        self.max_length = int(tokenizer.model_max_length)
        self.data_cfg = data_cfg

        hf_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )
        self.collator = BatchCollatorWithKeyAlignment(
            collator=hf_collator,
            keys_mapping={
                self.data_cfg.tokens_column: "input_ids",
                self.data_cfg.label_column: "labels",
            },
            max_length=self.data_cfg.max_length,
        )

        self.datasets_paths: dict[str, Path] | None = None

        self.train_dataset: HFDataset | None = None
        self.val_dataset: HFDataset | None = None
        self.test_dataset: HFDataset | None = None
        return

    @override
    def prepare_data(self) -> None:
        download_clearml_datasets(list(self.clearml_datasets.values()))
        return

    @staticmethod
    def _collect_split_paths(
        datasets_paths: dict[str, Path],
        split_name: str,
    ) -> dict[str, Path]:
        split_paths: dict[str, Path] = {}
        for dataset_name, dataset_path in datasets_paths.items():
            found_paths = list(dataset_path.glob(f"**/{split_name}/"))

            if len(found_paths) == 0:
                logger.warning(
                    f"No {split_name} data found in dataset {dataset_name} at path {dataset_path}."
                )
                continue

            for p in found_paths:
                split_paths[f"{dataset_name}/{p.parent.name}"] = p
        return split_paths

    @staticmethod
    def _concat_splits(
        split_paths: dict[str, Path], data_columns: set[str] | None = None
    ) -> HFDataset:
        datasets: list[HFDataset] = []
        for name, split_path in split_paths.items():
            ds = HFDataset.load_from_disk(split_path)
            if data_columns is not None:
                col2remove = []
                for col in ds.column_names:
                    if col not in data_columns:
                        col2remove.append(col)

                if len(col2remove) == len(ds.column_names):
                    logger.warning(
                        f"None of the specified data_columns {data_columns} found in {name}."
                    )
                    continue
                else:
                    ds = ds.remove_columns(col2remove)
            datasets.append(ds)  # type: ignore
        return concatenate_datasets(datasets)

    def _create_hf_dataset(
        self, split_name: Literal["train", "validation"]
    ) -> HFDataset:
        if self.datasets_paths is None:
            self.datasets_paths = get_datasets_paths(self.clearml_datasets)

        split_paths = self._collect_split_paths(self.datasets_paths, split_name)
        if len(split_paths) == 0:
            raise ValueError(
                f"No {split_name} data found in any of the provided datasets: {list(self.datasets_paths.keys())}"
            )

        dataset = self._concat_splits(
            split_paths,
            data_columns={
                self.data_cfg.tokens_column,
                self.data_cfg.label_column,
            },
        )
        return dataset

    @override
    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                if self.train_dataset is None:
                    self.train_dataset = self._create_hf_dataset(split_name="train")
                if self.val_dataset is None:
                    self.val_dataset = self._create_hf_dataset(split_name="validation")
            case "validate":
                if self.val_dataset is None:
                    self.val_dataset = self._create_hf_dataset(split_name="validation")
            case _:
                raise ValueError(f"Unsupported stage: {stage}")
        return

    @override
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_dataset is None:
            raise ValueError("Training dataset is not initialized.")
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized.")
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
