"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.3.1
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def show_data(raw_diabetes_data: pd.DataFrame) -> None:
    print("##### Dimensão #####")
    print(raw_diabetes_data.shape)
    print("##### Tipos #####")
    print(raw_diabetes_data.dtypes)
    print("##### Início #####")
    print(raw_diabetes_data.head())
    print("##### Final #####")
    print(raw_diabetes_data.tail())
    print("##### NA #####")
    print(raw_diabetes_data.isnull().sum())
    print("##### Quantiles #####")
    print(raw_diabetes_data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def detecting_categorical_numerical_variables(raw_diabetes_data: pd.DataFrame) -> None:
    df = raw_diabetes_data.copy()
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [
        col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes != "O"
    ]
    cat_but_car = [
        col for col in df.columns if df[col].nunique() > 20 and df[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observações: {df.shape[0]}")
    print(f"Variáveis: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")


def clean_data(
    raw_diabetes_data: pd.DataFrame, columns: dict[str, Any]
) -> pd.DataFrame:
    raw_df = raw_diabetes_data.copy()

    all_columns = [
        item
        for v in columns.values()
        for item in (v if isinstance(v, list) else [v])
        if item in raw_df.columns
    ]

    cleaned_df = raw_df[all_columns].copy()

    for c in columns["numerical"]:
        cleaned_df[c] = pd.to_numeric(cleaned_df[c], errors="coerce").fillna(0)

    logger.info(
        "Dados limpos: %d linhas, %d colunas", len(cleaned_df), len(cleaned_df.columns)
    )
    return cleaned_df
