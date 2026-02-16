"""Some tools for working with TensorDict."""

import numpy as np
import polars as pl
import torch
from funcy import first, is_list, omit, project
from tensordict import TensorDict, make_tensordict, merge_tensordicts, pad_sequence


def from_struct_array(arr: np.ndarray) -> TensorDict:
    """Convert a structured array to a tensordict, keeping types."""
    return make_tensordict(zip(arr.dtype.names, arr)).filter_non_tensor_data()


def cast(td: TensorDict, casts: dict):
    """Cast tensors in tensordict."""
    for k in td.keys():
        for src, dst in casts.items():
            if td[k].dtype == src:
                td[k] = td[k].to(dst)
    return td


def pl2td(df: pl.DataFrame):
    """Convert a polars dataframe to a list of tensordicts."""
    struct_np = df.to_numpy(structured=True, writable=True)
    return list(map(from_struct_array, struct_np))


def collate_pad_sequence(batch: tuple[dict, ...]):
    """Collate a sequence of polars rows into a batch, padding sequential features."""
    seq_keys = [k for k, v in batch[0].items() if is_list(v)]
    seq = [
        make_tensordict(project(b, seq_keys)).filter_non_tensor_data() for b in batch
    ]
    nonseq = [
        make_tensordict(omit(b, seq_keys)).filter_non_tensor_data() for b in batch
    ]

    padded_seq_feats: TensorDict = pad_sequence(seq, return_mask=True)
    padded_seq_feats["mask"] = first(padded_seq_feats["masks"].values())
    padded_seq_feats = padded_seq_feats.exclude("masks")
    padded_seq_feats.auto_batch_size_(1)

    # Nanify non-mask floating elements
    for k, v in padded_seq_feats.items():
        if v.dtype.is_floating_point:
            v[~padded_seq_feats["mask"]] = torch.nan

    stacked_cls_feats: TensorDict = torch.stack(nonseq)
    td = merge_tensordicts(padded_seq_feats, stacked_cls_feats)

    # somewhy td creates float64 by default
    return cast(td, {torch.float64: torch.float, torch.int64: torch.int})


def td2pl(td: TensorDict):
    """Convert a batched tensordict to a polars dataframe."""
    td = td.detach().cpu()
    np_dict: dict[list] = {k: list(v.numpy()) for k, v in td.items()}
    return pl.DataFrame(np_dict)
