"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.3.1
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler


def _feature_engineering_age(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    YOUNG = 21
    OLD = 50
    df["NEW_AGE_CAT"] = df["Age"].apply(
        lambda x: "senior" if x >= OLD else ("mature" if x >= YOUNG else "young")
    )
    return df


def _feature_engineering_bmi(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["NEW_BMI"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 24.9, 29.9, np.inf],
        labels=["Underweight", "Healthy", "Overweight", "Obese"],
    )
    return df


def _feature_engineering_glucose(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["NEW_GLUCOSE"] = pd.cut(
        df["Glucose"],
        bins=[0, 140, 200, np.inf],
        labels=["Normal", "Prediabetes", "Diabetes"],
    )
    return df


def _feature_engineering_age_bmi(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["NEW_AGE_BMI_NOM"] = (
        df["NEW_AGE_CAT"].astype(str) + "_" + df["NEW_BMI"].astype(str)
    )
    return df


def _feature_engineering_age_glucose(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["NEW_AGE_GLUCOSE_NOM"] = (
        df["NEW_AGE_CAT"].astype(str) + "_" + df["NEW_GLUCOSE"].astype(str)
    )
    return df


def _feature_engineering_insulin(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    LOWINSULIN = 16
    HIGHINSULIN = 166
    df["NEW_INSULIN_SCORE"] = df["Insulin"].apply(
        lambda x: "Abnormal" if (x < LOWINSULIN or x > HIGHINSULIN) else "Normal"
    )
    return df


def _feature_engineering_glucose_insulin(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["NEW_GLUCOSE_INSULIN"] = df["Glucose"] * df["Insulin"]
    return df


def create_features(outlier_handled_data: pd.DataFrame) -> pd.DataFrame:
    feature_engineered_data = outlier_handled_data.copy()
    feature_engineered_data = _feature_engineering_age(
        dataframe=feature_engineered_data
    )
    feature_engineered_data = _feature_engineering_bmi(
        dataframe=feature_engineered_data
    )
    feature_engineered_data = _feature_engineering_glucose(
        dataframe=feature_engineered_data
    )
    feature_engineered_data = _feature_engineering_age_bmi(
        dataframe=feature_engineered_data
    )
    feature_engineered_data = _feature_engineering_age_glucose(
        dataframe=feature_engineered_data
    )
    feature_engineered_data = _feature_engineering_insulin(
        dataframe=feature_engineered_data
    )
    feature_engineered_data = _feature_engineering_glucose_insulin(
        dataframe=feature_engineered_data
    )
    feature_engineered_data.columns = [
        c.upper() for c in feature_engineered_data.columns
    ]
    return feature_engineered_data


def encode_features(feature_engineered_data: pd.DataFrame) -> pd.DataFrame:
    encoded_data = feature_engineered_data.copy()
    target_col = "OUTCOME"

    binary_cols = [
        col
        for col in encoded_data.select_dtypes(include=["object", "category"]).columns
        if col != target_col and encoded_data[col].nunique() == 2  # noqa: PLR2004
    ]
    le = LabelEncoder()
    for col in binary_cols:
        encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))

    ohe_cols = [
        c
        for c in encoded_data.select_dtypes(include=["object", "category"]).columns
        if c != target_col and c not in binary_cols
    ]
    if ohe_cols:
        encoded_data = pd.get_dummies(encoded_data, columns=ohe_cols, drop_first=True)

    bool_cols = encoded_data.select_dtypes(include=["bool"]).columns
    encoded_data[bool_cols] = encoded_data[bool_cols].astype(int)
    return encoded_data


def scale_features(encoded_data: pd.DataFrame) -> tuple[pd.DataFrame, RobustScaler]:
    df = encoded_data.copy()
    target_col = "OUTCOME"
    numerical_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
    ]

    scaler = RobustScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler
