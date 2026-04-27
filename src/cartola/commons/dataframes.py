from collections.abc import Callable, Mapping

import pandas as pd


def drop_duplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
    result = df.loc[~df.duplicated()].copy()
    result.index = range(len(result))
    return result


def concat_partitioned_datasets(partitioned_dataset: Mapping[str, Callable[[], pd.DataFrame]]) -> pd.DataFrame:
    df_concat = pd.DataFrame()
    for _, partition_load_func in partitioned_dataset.items():
        partition_data = partition_load_func()
        df_concat = pd.concat([df_concat, partition_data], ignore_index=True)

    df_concat.index = range(len(df_concat))
    return df_concat


def rename_cols(df: pd.DataFrame, map_col_names: Mapping[str, str]) -> pd.DataFrame:
    return df.rename(columns=map_col_names)
