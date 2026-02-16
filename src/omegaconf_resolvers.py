"""File with our custom OmegaConf resolvers."""

from pathlib import Path

from omegaconf import OmegaConf


def register_path_resolvers():
    """Register some nice OmegaConf resolvers for path utilities."""
    OmegaConf.register_new_resolver("pathlib.stem", lambda path: Path(path).stem)
