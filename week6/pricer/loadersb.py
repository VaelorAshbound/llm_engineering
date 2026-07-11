import os
from typing import cast

from datasets import Dataset, concatenate_datasets, load_dataset
from pricer.parserb import is_valid, parse

WORKERS = max((os.cpu_count() or 2) - 1, 1)


def load_category(category: str, workers: int = WORKERS) -> Dataset:
    """Load one Amazon meta category and return curated, filtered rows."""
    dataset = cast(
        Dataset,
        load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
        ),
    )
    curated = dataset.map(
        parse,
        fn_kwargs={"category": category},
        num_proc=workers,
        remove_columns=dataset.column_names,
        desc=f"Parsing {category}",
    )
    return curated.filter(is_valid, num_proc=workers, desc=f"Filtering {category}")


def load_all(categories, workers: int = WORKERS) -> Dataset:
    """Load and concatenate every category into a single Dataset."""
    return concatenate_datasets(
        [load_category(category, workers) for category in categories]
    )
