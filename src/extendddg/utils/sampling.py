from __future__ import annotations

from typing import Optional, Tuple, Union

from beartype import beartype
from pandas import DataFrame


def _sample_rows(
    df: DataFrame, sample_size: int, random_state: int
) -> DataFrame | None:
    data_sample = None

    # Sample rows if sample_size is less than total rows
    if sample_size <= len(df):
        data_sample = df.sample(sample_size, random_state=random_state)

    return data_sample

def _sample_columns(
    df: DataFrame,
    sample_size: int,
    col_weights: Optional[Union[list[float], str]],
    random_state: int,
) -> DataFrame | None:
    data_sample = None

    if sample_size and sample_size < df.shape[1]:
        # Sample columns
        if col_weights:
            data_sample = df.sample(
                sample_size, axis=1, weights=col_weights, random_state=random_state
            )
        else:
            data_sample = df.sample(
                sample_size, axis=1, random_state=random_state
            )

        # Reorder columns to match original order
        original_order = [col for col in df.columns if col in data_sample.columns]
        data_sample = data_sample[original_order]

    return data_sample


@beartype
def get_sample(
    data_pd: DataFrame,
    row_sample_size: int,
    col_sample_size: Optional[int] = None,
    col_weights: Optional[Union[list[float], str]] = None,
    random_state: int = 9,
) -> Tuple[DataFrame, str]:
    """
    Return a random sample or whole frame and its CSV text

    Args:
        data_pd: Input DataFrame
        row_sample_size: Number of rows to sample
        col_sample_size: Number of columns to sample, optional
        col_weights: Weights for column sampling, optional
        random_state: Seed for reproducibility, optional

    Returns:
        (sample_df, sample_csv)
    """

    data_sample = _sample_rows(
        data_pd, sample_size=row_sample_size, random_state=random_state
    )

    # Perform column sampling if col_sample_size was provided
    if col_sample_size:
        # Set data_sample to full data if row sampling was not performed
        if data_sample is None:
            data_sample = data_pd

        data_sample = _sample_columns(
            data_sample,
            sample_size=col_sample_size,
            col_weights=col_weights,
            random_state=random_state,
        )

    # Set data_sample to full data if neither row or column sampling was performed
    if data_sample is None:
        data_sample = data_pd

    sample_csv = data_sample.to_csv(index=False)
    return data_sample, sample_csv
