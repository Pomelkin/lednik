from kostyl.ml.integrations.lightning import LightningCheckpointModelMixin
from transformers.modeling_utils import PreTrainedModel


class LednikPreTrainedModel(LightningCheckpointModelMixin, PreTrainedModel):
    """Base class for Lednik models."""

    pass
