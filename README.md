# Lednik

Lednik is a library for distilling large transformer models into lightweight, efficient representations—either **Static Embeddings** or small **Lednik Transformers**. It provides tools for model initialization, knowledge distillation training, and downstream classifier training.

## Features

- **Static Embeddings Distillation**: Compress a transformer's vocabulary into a static embedding matrix using PCA and Smooth Inverse Frequency (SIF) weighting.
- **Lednik Transformer Distillation**: Initialize a small specific transformer model ("Lednik") from a teacher model and distill knowledge into it.
- **Training Pipelines**: Ready-to-use [ClearML](https://clear.ml/) + [PyTorch Lightning](https://lightning.ai/) pipelines for distillation and classification.
- **Efficient Inference**: Optimized model classes for fast CPU/GPU inference.

## Project Structure

```
lednik/
├── distill/            # Distillation logic, factory methods, and training modules
├── models/             # Model definitions (LednikModel, StaticEmbeddingsModel)
├── serving/            # FastAPI serving utilities
pipelines/
├── finetuning/         # Knowledge Distillation pipeline
└── classifier_training/# Downstream classification pipeline
configs/                # Configuration files (YAML) for training
```

## Usage

### 1. Creating a Base Model

To start, you need to initialize a student model (Static or Lednik) from a pretrained Teacher transformer using the `model_factory`.

#### Static Embeddings

Extracts hidden states, applies PCA, and creates a `StaticEmbeddingsModel`.

```python
from transformers import AutoModel, AutoTokenizer
from lednik.distill.model_factory import create_static_embeddings_model

teacher = AutoModel.from_pretrained("deepvk/USER-bge-m3")
tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-bge-m3")

static_model = create_static_embeddings_model(
    model=teacher,
    tokenizer=tokenizer,
    embedding_dim=300,        # Target dimension
    pooling="mean",           # Pooling strategy: "mean", "cls", "last"
    embedding_extraction_batch_size=256,
    device="cuda"
)

static_model.save_pretrained("saved_models/static_base")
```

#### Lednik Transformer

Initializes a small transformer (`LednikModel`) compatible with the teacher's tokenizer.

```python
from lednik.models import LednikConfig
from lednik.distill.model_factory import create_lednik_transformer

config = LednikConfig(hidden_size=384, num_hidden_layers=2, ...)
lednik_model = create_lednik_transformer(
    model=teacher,
    tokenizer=tokenizer,
    model_config=config,
    pooling="cls",
    embedding_extraction_batch_size=256
)

lednik_model.save_pretrained("saved_models/lednik_base")
```

### 2. Knowledge Distillation

The distillation process aligns the student model's embeddings with the teacher's on specific datasets. This uses the `pipelines/finetuning/run.py` script.

#### Configuration

The training pipeline uses hardcoded paths to configuration files in the `configs/finetuning/` directory. You should edit these files before running the script:

1.  **Distillation Config** (`configs/finetuning/distill_config.yaml`): Defines hyperparameters like Learning Rate, Optimizer, and Loss weights.
    ```yaml
    optimizer:
      type: AdamW8bit
      betas: [0.9, 0.98]
    lr:
      scheduler_type: plateau-with-cosine-annealing
      base_value: 9e-5
    distillation_method:
      type: direct-distillation
      loss_type: cosine
      contrastive_loss_weight: 0.8  # Weight for contrastive loss
      temperature: 0.07             # Temp for contrastive loss
    batch_size: 96
    num_workers: 4
    ```

2.  **Training Settings** (`configs/finetuning/training_settings.yaml`): Defines the environment, data paths, and trainer settings.
    ```yaml
    data:
      train_datasets:
        Dataset1: <CLEARML_ID>
    trainer:
      accelerator: "cuda"
      devices: [0]
      strategy: "fsdp1"
      max_epochs: 2
    # ClearML IDs for artifacts
    teacher_model_id: <TEACHER_ID>
    student_model_id: <STUDENT_ID>  # The model created in step 1
    tokenizer_id: <TOKENIZER_ID>
    ```

#### Running the Training

Run the pipeline module directly. The script automatically picks up the configuration files from the `configs/` directory.

```bash
python -m pipelines.finetuning.run
```

Optional arguments:
- `--remote-execution-queue <QUEUE_NAME>`: Run the task remotely on a ClearML queue.
- `--tags tag1,tag2`: Add tags to the ClearML task.

### 3. Training a Classifier

Once you have a distilled model, you can fine-tune it for a specific downstream classification task using `pipelines/classifier_training/run.py`.

#### Configuration

Edit the configuration files in `configs/classification/`:

1.  **Training Config** (`configs/classification/train_config.yaml`):
    ```yaml
    label2id:
      negative: 0
      neutral: 1
      positive: 2
    lr:
      base_value: 7e-4
    class_weights: [0.55, 0.24, 0.21] # Optional
    batch_size: 128
    num_workers: 8
    ```

2.  **Training Settings** (`configs/classification/training_settings.yaml`):
    ```yaml
    data:
      datasets:
        MyDataset: <CLEARML_ID>
      label_column: label
      input_column: text
    model_id: <DISTILLED_MODEL_ID> # Result from step 2
    trainer:
      accelerator: "cuda"
      max_epochs: 10
    ```

#### Running the Training

```bash
python -m pipelines.classifier_training.run
```

Optional arguments:
- `--remote-execution-queue <QUEUE_NAME>`: Run the task remotely on a ClearML queue.
- `--tags tag1,tag2`: Add tags to the ClearML task.

## Models

### LednikModel
A lightweight transformer model designed to be initialized from a larger parent model. It supports features like Rotary Positional Embeddings (RoPE) and Flash Attention.

### StaticEmbeddingsModel
A simple model that maps tokens directly to static vectors, optionally handling subword pooling and normalization. Ideal for extremely low-latency scenarios where full transformer context is not strictly required.
