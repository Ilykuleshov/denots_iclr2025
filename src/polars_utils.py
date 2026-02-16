"""File with preprocessing utils based on the Polars library."""

import random

import polars as pl
from polars import selectors as cs


def frequency_encode(expr: pl.Expr, clip: bool = True):
    """Modify a column in a Polars DataFrame by replacing certain values with a new, more manageable set of categories.

    The function works by first counting the frequency of each unique value in the column.
    It sorts these values in descending order of frequency and keeps only those that make up a certain percentage of the
    total. The default percentage is 95%, but this can be adjusted by changing the quantile argument.

    Arguments:
    ---------
    expr:
        This is a Polars Expr object, which represents a column in a DataFrame. The function will modify this column by
        replacing certain values with a new, more manageable set of categories.
    clip:
        whether to perform clipping.


    """
    nonnull_expr = expr.filter(expr.is_not_null())
    value_counts = nonnull_expr.value_counts(sort=True, normalize=True, parallel=True)
    expr = expr.replace(
        value_counts.struct[0], pl.int_range(1, value_counts.len() + 1)
    ).cast(pl.Int32)

    if clip:
        fracs = value_counts.struct[1]
        remainder = fracs.cum_sum(reverse=True) - fracs
        is_major = fracs.first() < remainder
        expr = expr.clip(upper_bound=is_major.sum() + 2)

    return expr


def random_split(
    df: pl.LazyFrame | pl.DataFrame,
    seed: int | None = 42,
    alias: str = "split",
    **splits: dict[str, int | float | None],
):
    """Split a DataFrame into multiple parts based on the provided split ratios.

    Arguments:
    ---------
        df:
            A pl.LazyFrame or pl.DataFrame object.
        seed: An optional integer to set the random seed. Defaults to 42.
        alias: An optional string to set the alias for the split column. Defaults to "split".
        **splits:
            A variable-length dictionary of split ratios. Each key-value pair represents a split.
            The is the name of the split, and the value is either a float or an integer representing the
            split ratio. If the value is a float, it is interpreted as a proportion of the total number of rows
            in the DataFrame. If the value is an integer, it is interpreted as an absolute number of rows.
            If a value is floating, it represents the remaining rows after all other splits have been applied.

    """
    if not splits:
        raise ValueError("Provide at least one split!")

    df = df.with_row_index().with_columns(pl.col("index").shuffle(seed=seed))

    index_max = 0
    remaining = None
    expr = pl
    for k, v in splits.items():
        if v is None:
            if remaining is not None:
                raise ValueError("More than one None argument!")
            remaining = k
            continue

        if isinstance(v, float):
            v = pl.col("index").max() * v

        expr = expr.when(pl.col("index") < v + index_max).then(pl.lit(k))
        index_max = index_max + v

    remaining = remaining or k
    expr = expr.otherwise(pl.lit(remaining))
    return df.with_columns(expr.alias(alias)).drop("index")


def normalize(expr: pl.Expr):
    """Normalize given expression."""
    return (expr - expr.drop_nans().mean()) / expr.drop_nans().std()


def cum_diff(expr: pl.Expr):
    """Count cumulative differences in expr (could be multicol)."""
    return pl.any_horizontal((expr != expr.shift()).fill_null(True)).cum_sum()


def nanify(expr: pl.Expr, frac: float):
    """Convert a random fraction of the input list to nans."""
    return expr.map_elements(
        lambda x: x if random.random() > frac else float("nan"), return_dtype=pl.Float64
    )


def filter_horizontal(df: pl.DataFrame, expr: pl.Expr):
    """Filter dataframe's columsn according to an expression."""
    return df.select(c.name for c in df.select(expr) if c.all())


def best_masks(expr: pl.Expr, frac_std=0.5):
    mean = expr.list.get(0).cast(float)
    std = expr.list.get(1).cast(float)
    mask = (mean.max() - mean) < (
        (std**2 + std.get(mean.arg_max()) ** 2) ** 0.5
    ) * frac_std

    mean2 = pl.when(mask.not_()).then(mean)
    mask2 = (
        mean2.max() - mean2
        < ((std**2 + std.get(mean2.arg_max()) ** 2) ** 0.5) * frac_std
    )

    return mask, mask2


FLOATLIST = frozenset(map(pl.List, cs.FLOAT_DTYPES))
INTLIST = frozenset(map(pl.List, cs.INTEGER_DTYPES))
TIMELIST = frozenset(map(pl.List, cs.TEMPORAL_DTYPES))
NUMLIST = FLOATLIST | INTLIST
