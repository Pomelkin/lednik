from functools import partial
from pathlib import Path
from typing import Literal
from typing import override

import lightning as L
import torch
from kostyl.ml_core.clearml.dataset_utils import collect_clearml_datasets
from kostyl.ml_core.clearml.dataset_utils import download_clearml_datasets
from kostyl.ml_core.clearml.dataset_utils import get_datasets_paths
from kostyl.utils.logging import setup_logger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .config import DataConfig
from datasets import concatenate_datasets
from datasets import Dataset as HFDataset


logger = setup_logger(fmt="only_message")


class DataModule(L.LightningDataModule):
    """A DataModule standardizes the training, val, test splits, data preparation and transforms."""

    def __init__(
        self,
        data_cfg: DataConfig | dict,
        tokenizer: PreTrainedTokenizerBase,
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
            tokenizer (PreTrainedTokenizerBase): The tokenizer instance used for processing
                text data.

        """
        super().__init__()
        if isinstance(data_cfg, dict):
            data_cfg = DataConfig.model_validate(data_cfg)
        self.save_hyperparameters({"data_cfg": data_cfg.model_dump()})

        self.train_clearml_datasets = collect_clearml_datasets(data_cfg.train_datasets)
        self.val_clearml_datasets = collect_clearml_datasets(data_cfg.val_datasets)
        self.all_clearml_datasets = [
            *self.train_clearml_datasets.values(),
            *self.val_clearml_datasets.values(),
        ]
        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.num_workers
        self.train_tokens_column = data_cfg.train_tokens_column
        self.val_tokens_column = data_cfg.val_tokens_column
        self.val_label_column = data_cfg.val_label_column

        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id)  # pyright: ignore[reportArgumentType]
        self.max_length = int(tokenizer.model_max_length)  # pyright: ignore[reportArgumentType]
        self.data_cfg = data_cfg

        self.train_datasets_paths: dict[str, Path] | None = None
        self.val_datasets_paths: dict[str, Path] | None = None

        self.train_dataset: HFDataset | None = None
        self.val_dataset: HFDataset | None = None
        self.test_dataset: HFDataset | None = None
        return

    @override
    def prepare_data(self) -> None:
        download_clearml_datasets(self.all_clearml_datasets)
        return

    @staticmethod
    def _collect_split_paths(
        datasets_paths: dict[str, Path], split_name: str
    ) -> dict[str, Path]:
        split_paths: dict[str, Path] = {}
        for name, path in datasets_paths.items():
            found_paths = list(path.glob(f"**/{split_name}/"))

            if len(found_paths) == 0:
                logger.warning(
                    f"No {split_name} data found in dataset {name} at path {path}."
                )
                continue

            for p in found_paths:
                split_paths[f"{name}/{p.parent.name}"] = p
        return split_paths

    @staticmethod
    def _concat_splits(
        split_paths: dict[str, Path], data_columns: set[str] | None = None
    ) -> HFDataset:
        datasets = []
        for name, split_path in split_paths.items():
            ds = HFDataset.load_from_disk(split_path)

            if data_columns is not None:
                col2remove = []
                for col in ds.column_names:
                    if col not in data_columns:
                        col2remove.append(col)

                if len(col2remove) == len(ds.column_names):
                    raise ValueError(
                        f"None of the specified data_columns {data_columns} found in dataset {name}."
                    )
                ds = ds.remove_columns(col2remove)

            datasets.append(ds)
        return concatenate_datasets(datasets)

    def _create_hf_dataset(self, dataset_type: Literal["train", "val"]) -> HFDataset:
        match dataset_type:
            case "train":
                if self.train_datasets_paths is None:
                    self.train_datasets_paths = get_datasets_paths(
                        self.train_clearml_datasets
                    )
                datasets_paths = self.train_datasets_paths
                data_columns = {self.train_tokens_column}
                split_name = "train"
            case "val":
                if self.val_datasets_paths is None:
                    self.val_datasets_paths = get_datasets_paths(
                        self.val_clearml_datasets
                    )
                datasets_paths = self.val_datasets_paths
                data_columns = {self.val_tokens_column, self.val_label_column}
                split_name = "validation"
            case _:
                raise ValueError(f"Unknown dataset_type: {dataset_type}")

        split_paths = self._collect_split_paths(datasets_paths, split_name)

        if len(split_paths) == 0:
            raise ValueError(
                f"No {split_name} data found in any of the provided datasets: {list(datasets_paths.keys())}"
            )

        dataset = self._concat_splits(split_paths, data_columns=data_columns)
        return dataset

    @override
    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                if self.train_dataset is None:
                    self.train_dataset = self._create_hf_dataset(dataset_type="train")
                if self.val_dataset is None:
                    self.val_dataset = self._create_hf_dataset(dataset_type="val")
            case "validate":
                if self.val_dataset is None:
                    self.val_dataset = self._create_hf_dataset(dataset_type="val")
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
            collate_fn=partial(
                train_collator,
                pad_token_id=self.pad_token_id,
                max_length=self.max_length,
                tokens_key=self.train_tokens_column,
            ),
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
            collate_fn=partial(
                val_collator,
                pad_token_id=self.pad_token_id,
                tokens_key=self.val_tokens_column,
                label_key=self.val_label_column,
                max_length=self.max_length,
            ),
        )


def train_collator(
    batch: list[dict[str, list[int]]],
    pad_token_id: int,
    max_length: int,
    tokens_key: str,
) -> dict[str, torch.Tensor]:
    """
    Collates a batch of examples into padded tensors for training.

    Args:
        batch (list[dict[str, list[int]]]): A list of examples, where each example
            is a dictionary containing the token IDs under `tokens_key`.
        pad_token_id (int): The integer ID used for padding sequences.
        max_length (int): The maximum allowed sequence length. Sequences longer
            than this will be truncated.
        tokens_key (str): The key in the input dictionaries where the token ID
            list is stored.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing:
            - "input_ids": A tensor of shape (batch_size, sequence_length) containing
                padded token IDs.
            - "attention_mask": A tensor of shape (batch_size, sequence_length)
                containing 1s for valid tokens and 0s for padding tokens.

    """
    input_ids_list = [
        torch.tensor(item[tokens_key], dtype=torch.long) for item in batch
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=pad_token_id,
    )
    input_ids = input_ids[:, :max_length]
    attention_mask = (input_ids != pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def val_collator(
    batch: list[dict[str, list[int] | int]],
    pad_token_id: int,
    tokens_key: str,
    max_length: int,
    label_key: str,
) -> dict[str, torch.Tensor]:
    """
    Collates a batch of validation data into padded tensors.

    Args:
        batch (list[dict[str, list[int] | int]]): A list of data items, where each item is a
            dictionary containing token IDs and labels.
        pad_token_id (int): The integer ID used for padding sequences.
        tokens_key (str): The dictionary key used to access token IDs in the batch items.
        label_key (str): The dictionary key used to access labels in the batch items.
        max_length (int): The maximum allowed sequence length. Sequences longer
            than this will be truncated.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the collated batch with keys:
            - "input_ids": Padded tensor of input token IDs (batch_size, max_seq_len).
            - "attention_mask": Tensor indicating non-padded elements (batch_size, max_seq_len).
            - "labels": Tensor of labels (batch_size,).

    """
    input_ids_list: list[torch.Tensor] = []
    labels_list: list[int] = []
    for item in batch:
        input_ids_list.append(torch.tensor(item[tokens_key], dtype=torch.long))
        labels_list.append(item[label_key])  # type: ignore

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=pad_token_id,
    )
    input_ids = input_ids[:, :max_length]
    labels = torch.tensor(labels_list, dtype=torch.long)
    attention_mask = (input_ids != pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
