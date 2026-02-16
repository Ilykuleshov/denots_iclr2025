"""Some utilities for working with nn.Modules."""

from contextlib import contextmanager
from types import MethodType

from torch import nn


@contextmanager
def freeze_module(module):
    """Freeze a given nn.Module.

    This disables grads and puts the module in constant eval mode.
    """
    module.train(False)
    module.train = MethodType(lambda self, mode: self, module)
    module.requires_grad_(False)
    yield module

    module.train = MethodType(nn.Module.train, module)
    module.train(True)
    module.requires_grad_(True)


@contextmanager
def disable_module(module):
    """Disable a TensorDict-like module, turning it into a dummy.

    This replaces the forward function a dummy, and sets
    the in/out keys accordingly..
    """
    module._old_forward = module.forward  # type: ignore
    module.forward = MethodType(lambda self, *args, **kwargs: None, module)
    if hasattr(module, "out_keys"):
        module._old_out_keys = module.out_keys
        module.out_keys = ["_"]  # type: ignore

    yield module
    module.forward = module._old_forward
    if hasattr(module, "_old_out_keys"):
        module.out_keys = module._old_out_keys


ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
}
