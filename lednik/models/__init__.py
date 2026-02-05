from .configuration_lednik import LednikConfig
from .configuration_static import StaticEmbeddingsConfig
from .modeling_lednik import LednikModel
from .modeling_static import StaticEmbeddingsForSequenceClassification
from .modeling_static import StaticEmbeddingsModel
from .outputs import StaticEmbeddingsOutput
from .outputs import StaticEmbeddingsSequenceClassifierOutput


MODEL_MAPPING: dict[str, type[LednikModel] | type[StaticEmbeddingsModel]] = {
    "lednik": LednikModel,
    "static_embeddings": StaticEmbeddingsModel,
}

CONFIG_MAPPING: dict[str, type[LednikConfig] | type[StaticEmbeddingsConfig]] = {
    "lednik": LednikConfig,
    "static_embeddings": StaticEmbeddingsConfig,
}

CONFIG_TO_MODEL_MAPPING: dict[
    type[LednikConfig] | type[StaticEmbeddingsConfig],
    type[LednikModel] | type[StaticEmbeddingsModel],
] = {
    LednikConfig: LednikModel,
    StaticEmbeddingsConfig: StaticEmbeddingsModel,
}


__all__ = [
    "CONFIG_MAPPING",
    "CONFIG_TO_MODEL_MAPPING",
    "MODEL_MAPPING",
    "LednikConfig",
    "LednikModel",
    "StaticEmbeddingsConfig",
    "StaticEmbeddingsForSequenceClassification",
    "StaticEmbeddingsModel",
    "StaticEmbeddingsOutput",
    "StaticEmbeddingsSequenceClassifierOutput",
]
