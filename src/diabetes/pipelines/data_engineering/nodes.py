"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.3.1
"""

import itertools
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

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
    _analysing_categorical_numerical_variables(raw_diabetes_data=raw_diabetes_data)
    _correlation_analysis(raw_diabetes_data=raw_diabetes_data)
    _relationship_between_variables(raw_diabetes_data=raw_diabetes_data)


def _cat_summary(dataframe, col_name, plot=False):
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


def _num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    logger.info(dataframe[col_name].describe(quantiles).T)
    if plot:
        dataframe[col_name].hist(bins=20)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)


def _target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    logger.info(
        pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()})
    )
    if plot:
        sns.barplot(x=categorical_col, y=target, data=dataframe)
        plt.show(block=True)


def _target_summary_with_num(dataframe, target, numerical_col, plot=False):
    logger.info(
        pd.DataFrame(
            {numerical_col + "_mean": dataframe.groupby(target)[numerical_col].mean()}
        )
    )
    if plot:
        sns.barplot(x=target, y=numerical_col, data=dataframe)
        plt.show(block=True)


def _analysing_categorical_numerical_variables(raw_diabetes_data: pd.DataFrame) -> None:
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
    logger.info(_cat_summary(raw_df, "Outcome", plot=True))
    for col in cat_cols:
        _cat_summary(raw_df, col, plot=True)
    for col in num_cols:
        _num_summary(raw_df, col, plot=True)
    for col in cat_cols:
        _target_summary_with_cat(raw_df, "Outcome", col, plot=True)
    for col in num_cols:
        _target_summary_with_num(raw_df, "Outcome", col, plot=True)


def _diabetes_heatmap(corr: pd.DataFrame, corr_values: pd.DataFrame) -> None:
    sns.set_theme(rc={"figure.figsize": (12, 12)})
    sns.heatmap(corr, cmap="RdBu", annot=corr_values)
    plt.show(block=True)


def _diabetes_high_correlated_cols(
    dataframe: pd.DataFrame, plot=False, corr_th=0.7
) -> list[str]:
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)
    )
    drop_list = [
        col
        for col in upper_triangle_matrix.columns
        if any(upper_triangle_matrix[col] > corr_th)
    ]
    if plot:
        sns.set_theme(rc={"figure.figsize": (12, 12)})
        corr_values = corr.round(2)
        sns.heatmap(corr, cmap="RdBu", annot=corr_values)
        plt.show(block=True)
    return drop_list


def _correlation_analysis(raw_dibetes_data: pd.DataFrame) -> None:
    raw_df = raw_dibetes_data.copy()
    corr = raw_df.corr()
    logger.info(corr)
    corr_values = corr.round(2)
    _diabetes_heatmap(corr=corr, corr_values=corr_values)
    _diabetes_high_correlated_cols(dataframe=raw_df, plot=True)


def _relationship_between_variables(raw_dibetes_data: pd.DataFrame) -> None:
    raw_df = raw_dibetes_data.copy()
    outcome_counts = raw_df["Outcome"].value_counts()

    total_patients = outcome_counts.sum()

    percentages = outcome_counts / total_patients * 100

    labels = [
        f"0 - Nao-Diabeticos\n({outcome_counts[0]} / {percentages[0]:.1f}%)",
        f"1 - Diabeticos\n({outcome_counts[1]} / {percentages[1]:.1f}%)",
    ]

    plt.figure(figsize=(8, 6))
    plt.pie(
        outcome_counts, labels=labels, autopct="%1.1f%%", colors=["purple", "lightgray"]
    )
    plt.title("Distribuicao da variavel Outcome")
    plt.show()

    sns.pairplot(
        data=raw_df,
        vars=[
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        hue="Outcome",
        height=5,
    )
    plt.show(block=True)

    feature_combinations = list(
        itertools.combinations(
            [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ],
            2,
        )
    )

    for i, (feature1, feature2) in enumerate(feature_combinations):
        fig = px.scatter(
            raw_df,
            x=feature1,
            y=feature2,
            color="Outcome",
            size="BMI",
            title=f"{feature1} vs {feature2} Bubble Chart",
        )
        fig.show()


def _missing_values(raw_diabetes_data: pd.DataFrame) -> None:
    df = raw_diabetes_data.copy()
    zero_columns = [
        col
        for col in df.columns
        if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])
    ]
    for col in zero_columns:
        df[col] = np.where(df[col] == 0, np.nan, df[col])
    logger.info(df.isnull().sum())


def _outlier_thresholds(
    dataframe: pd.DataFrame, col_name: str, q1: float = 0.05, q3: float = 0.95
) -> tuple[float, float]:
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def _has_outliers(dataframe: pd.DataFrame, col_name: str) -> bool:
    low_limit, up_limit = _outlier_thresholds(dataframe, col_name)

    return ((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)).any()


def _check_outliers(raw_diabetes_data: pd.DataFrame) -> None:
    df = raw_diabetes_data.copy()

    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            logger.info("%s: skipped (non-numeric)", col)
            continue

        has_outlier = _has_outliers(df, col)
        logger.info("%s: %s", col, has_outlier)


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
        cleaned_df[c] = pd.to_numeric(cleaned_df[c], errors="coerce")

    logger.info(
        "Dados limpos: %d linhas, %d colunas", len(cleaned_df), len(cleaned_df.columns)
    )
    return cleaned_df


def check_missing_outliers_values(cleaned_data: pd.DataFrame) -> None:
    df = cleaned_data.copy()
    logger.info("\nDados faltantes")
    _missing_values(raw_diabetes_data=df)
    logger.info("\nAnomalias")
    _check_outliers(raw_diabetes_data=df)


def handle_missing_values(
    cleaned_diabetes_data: pd.DataFrame, missing_values: dict[str, Any]
) -> pd.DataFrame:
    df_handled_missing_values = cleaned_diabetes_data.copy()
    zero_columns = missing_values["zero_columns"]
    n_neighbors = missing_values["knn_neighbors"]

    for col in zero_columns:
        if col in df_handled_missing_values.columns:
            df_handled_missing_values[col] = df_handled_missing_values[col].replace(
                0, np.nan
            )

    target_col = "Outcome"
    feature_cols = [c for c in df_handled_missing_values.columns if c != target_col]

    df_handled_missing_values[feature_cols] = df_handled_missing_values[feature_cols].astype(float)

    scaler = RobustScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_handled_missing_values[feature_cols]),
        columns=feature_cols,
        index=df_handled_missing_values.index,
    )

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_scaled),
        columns=feature_cols,
        index=df_handled_missing_values.index,
    )

    df_handled_missing_values[feature_cols] = scaler.inverse_transform(df_imputed)
    return df_handled_missing_values


def handle_outliers(
    df_without_missing_values: pd.DataFrame, outliers: dict[str, float]
) -> pd.DataFrame:
    df_without_outilers = df_without_missing_values.copy()
    q1_pct = outliers["q1"]
    q3_pct = outliers["q3"]
    target_col = "Outcome"
    numerical_cols = [
        c
        for c in df_without_outilers.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    for col in numerical_cols:
        low = df_without_outilers[col].quantile(q1_pct)
        high = df_without_outilers[col].quantile(q3_pct)
        df_without_outilers[col] = df_without_outilers[col].clip(lower=low, upper=high)

    return df_without_outilers
