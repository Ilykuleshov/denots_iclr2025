"""Module with the DataSource class.

This class is used to load data from a .parquet file.
"""

import polars as pl
from funcy import omit, select_values
from polars import selectors as cs

from .iterfunctools import sort_keys
from .polars_utils import FLOATLIST, INTLIST


class DataSource:
    """The DataSource class for loading data.

    This handles several things:
     - loads data from a .parquet file,
     - preprocesses and caches it on call,
     - creates the data schema, describing all present columns.
    """

    def __init__(self, source: str):
        """Initialize the DataSource class.

        Args:
        ----
            source (str): Path to the .parquet file.

        """
        self.source = source

    def __call__(self):
        """Scan the data, returning a LazyFrame."""
        if hasattr(self, "df"):
            return self.df

        df = pl.scan_parquet(self.source).drop(
            cs.string().exclude("split"), cs.temporal()
        )

        df = df.with_row_index().with_columns(
            cs.integer().cast(pl.Int32),
            cs.float().cast(pl.Float32),
            cs.by_dtype(FLOATLIST).cast(pl.List(pl.Float32)),
            cs.by_dtype(INTLIST).cast(pl.List(pl.Int32)),
        )

        self.df = df.cache()
        return self.df

    def get_schema(self):
        """Gather the metadata about the stored datasets."""
        if hasattr(self, "_schema"):
            return self._schema

        df = self().drop("time", "split", "index")
        pl_schema = df.collect_schema()

        schema = {}
        if pl.List(pl.Int32) in pl_schema.values():
            cat_feats = (
                df.select(pl.col(INTLIST).explode().max() + 1).collect().to_dicts()
            )[0]
        else:
            cat_feats = {}

        if pl.Int32 in pl_schema.values():
            cat_labels = (
                df.select(cs.integer() - cs.binary()).max().collect() + 1
            ).to_dicts()[0]
        else:
            cat_labels = {}

        schema["category"] = sort_keys(cat_feats)
        schema["numeric"] = sorted(select_values(FLOATLIST, pl_schema).names())

        schema["label"] = {
            "category": sort_keys(cat_labels),
            "numeric": sorted(select_values(cs.FLOAT_DTYPES, pl_schema).names()),
            "binary": sorted(select_values({pl.Boolean}, pl_schema).names()),
        }

        self._schema = schema
        return schema

    def get_split_counts(self):
        """Count rows for each split."""
        return dict(
            self()
            .select(pl.col("split").value_counts())
            .unnest("split")
            .collect()
            .iter_rows()
        )

    @staticmethod
    def label_type(schema: dict, label_name: str):
        """Determine the type of a given label."""
        label_schema = schema["label"]

        if "target" in label_schema["category"]:
            return "category"
        elif "target" in label_schema["binary"]:
            return "binary"
        elif "target" in label_schema["numeric"]:
            return "numeric"
        else:
            raise ValueError(
                f"Unable to determine supervision type: '{label_name}' not found in the label schema."
            )

    @staticmethod
    def feature_type(schema: dict, feature_name: str):
        """Return the type of the given feature, based on the provided schema."""
        if feature_name in schema["category"]:
            return "category"
        if feature_name in schema["numeric"]:
            return "numeric"

        raise ValueError(f"feature {feature_name} not found!")

    @staticmethod
    def labels(schema: dict):
        """Return the labels of the given data."""
        return [lbl for lblcoll in schema["label"].values() for lbl in lblcoll]

    @staticmethod
    def features(schema: dict):
        """Return the features of the given data."""
        return [feat for coll in omit(schema, ("label",)).values() for feat in coll]
