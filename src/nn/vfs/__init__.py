"""The module for the various Vector Fields for our project."""

from .interprnn import (
    AdaptiveGRUInterpVF,
    ReLUInterpVF,
    StrictGRUInterpVF,
    TanhInterpVF,
)

INTERP_VFS: dict = {
    "adaptive": AdaptiveGRUInterpVF,
    "strict": StrictGRUInterpVF,
    "tanh": TanhInterpVF,
    "relu": ReLUInterpVF,
}

__all__ = ["AdaptiveGRUInterpVF", "StrictGRUInterpVF", "TanhInterpVF", "ReLUInterpVF"]
