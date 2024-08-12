from .example_cnn_classifier import ExampleCNN

# rethink the location of this module. If we're not careful, we end up with circular imports
# when using the `fibad_model` decorator on models in this module.
from .model_registry import MODEL_REGISTRY, fibad_model

__all__ = ["fibad_model", "MODEL_REGISTRY", "ExampleCNN"]
