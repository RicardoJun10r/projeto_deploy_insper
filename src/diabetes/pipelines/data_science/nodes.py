"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.3.1
"""

import logging
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def split_data(
    model_input_data: pd.DataFrame,
    split: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_col = "OUTCOME"
    X = model_input_data.drop(columns=[target_col])
    y = model_input_data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split["test"], random_state=split["random_state"]
    )
    logger.info("Split: train=%d, test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train.to_frame(), y_test.to_frame()


_MODEL_REGISTRY: dict[str, type[ClassifierMixin]] = {
    "rf": RandomForestClassifier,
    "lr": LogisticRegression,
    "knn": KNeighborsClassifier,
    "svc": SVC,
    "dt": DecisionTreeClassifier,
    "ada": AdaBoostClassifier,
    "gbm": GradientBoostingClassifier,
    "xgb": XGBClassifier,
    "lgbm": LGBMClassifier,
}

_METRIC_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "Accuracy": lambda y_true, y_pred, _: accuracy_score(y_true, y_pred),
    "Recall": lambda y_true, y_pred, _: recall_score(y_true, y_pred),
    "Precision": lambda y_true, y_pred, _: precision_score(y_true, y_pred),
    "F1": lambda y_true, y_pred, _: f1_score(y_true, y_pred),
    "AUC": lambda y_true, _, y_prob: roc_auc_score(y_true, y_prob),
}


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    models: dict[str, Any],
    model_params: dict[str, Any],
) -> dict[str, ClassifierMixin]:
    y: pd.Series = y_train.iloc[:, 0]
    trained: dict[str, ClassifierMixin] = {}
    for name in models["names"]:
        params: dict[str, Any] = model_params.get(name, {})
        model: ClassifierMixin = _MODEL_REGISTRY[name](**params)
        model.fit(X_train, y)
        trained[name] = model
        logger.info("Modelo treinado: %s (params: %s)", name, params)
    return trained


def evaluate_models(
    trained_models: dict[str, ClassifierMixin],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    models: dict[str, Any],
) -> pd.DataFrame:
    y_true: pd.Series = y_test.iloc[:, 0]
    results: list[dict[str, Any]] = []
    for name, model in trained_models.items():
        y_pred: np.ndarray = model.predict(X_test)
        y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]
        row: dict[str, Any] = {"Model": name}
        for metric in models["metrics"]:
            row[metric] = round(_METRIC_REGISTRY[metric](y_true, y_pred, y_prob), 4)
        results.append(row)
        logger.info("Métricas [%s]: %s", name, row)
    return pd.DataFrame(results)


def plot_model_metrics(model_metrics: pd.DataFrame) -> None:
    logger.info("\n%s", model_metrics.to_string(index=False))

    HEIGHT_LIMIT_BAR: float = 0.55
    df: pd.DataFrame = model_metrics.sort_values(by="AUC", ascending=False)
    metrics: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()

    df_melted: pd.DataFrame = df.melt(
        id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score"
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        order=metrics,
        dodge=True,
        width=0.75,
    )

    for container in ax.containers:
        labels: list[str] = [
            f"{bar.get_height():.3f}" if bar.get_height() >= HEIGHT_LIMIT_BAR else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, fontsize=8, padding=2)

    ax.set_title(
        "Comparação dos Modelos (com Feature Engineering)",
        fontsize=16,
        weight="bold",
        pad=15,
    )
    ax.set_xlabel("Métrica", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=0)
    plt.legend(
        title="Modelo", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine()
    plt.tight_layout()
    plt.show()
