from .registry import (
    DatasetAdapter,
    DATASET_REGISTRY,
    get_adapter,
    list_adapter_keys,
    list_adapters,
    register_adapter,
)

# Register built-in adapters
from . import ashrae  # noqa: F401
from . import pronto  # noqa: F401
from . import swat    # noqa: F401

__all__ = [
    "DatasetAdapter",
    "DATASET_REGISTRY",
    "get_adapter",
    "list_adapter_keys",
    "list_adapters",
    "register_adapter",
]
