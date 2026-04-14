"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.3.1
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def show_data(raw_diabetes_data: pd.DataFrame) -> None:
    logger.info("##### Dimensão #####")
    logger.info(raw_diabetes_data.shape)
    logger.info("##### Tipos #####")
    logger.info(raw_diabetes_data.dtypes)
    logger.debug("##### Início #####")
    logger.debug(raw_diabetes_data.head())
    logger.debug("##### Final #####")
    logger.debug(raw_diabetes_data.tail())
    logger.info("##### NA #####")
    logger.info(raw_diabetes_data.isnull().sum())
    logger.info("##### Quantiles #####")
    logger.info(raw_diabetes_data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    logger.info(
        pd.DataFrame(
            {
                col_name: dataframe[col_name].value_counts(),
                "Razao": 100 * dataframe[col_name].value_counts() / len(dataframe),
            }
        )
    )
    logger.info(80 * "=")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def detecting_categorical_numerical_variables(raw_diabetes_data: pd.DataFrame) -> None:
    raw_df = raw_diabetes_data.copy()
    NUM_CAT = 10
    CAT_CAR = 20
    cat_cols = [col for col in raw_df.columns if raw_df[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in raw_df.columns
        if raw_df[col].nunique() < NUM_CAT and raw_df[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in raw_df.columns
        if raw_df[col].nunique() > CAT_CAR and raw_df[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in raw_df.columns if raw_df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    logger.info(f"Observações: {raw_df.shape[0]}")
    logger.info(f"Variáveis: {raw_df.shape[1]}")
    logger.info(f"cat_cols: {len(cat_cols)}")
    logger.info(f"num_cols: {len(num_cols)}")
    logger.info(f"cat_but_car: {len(cat_but_car)}")
    logger.info(f"num_but_cat: {len(num_but_cat)}")
    logger.info(80 * "=")
    logger.info(cat_summary(raw_df, "Outcome", plot=True))
    for col in cat_cols:
        cat_summary(raw_df, col, plot=True)


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
