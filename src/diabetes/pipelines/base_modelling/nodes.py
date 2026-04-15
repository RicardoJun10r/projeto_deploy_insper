"""
This is a boilerplate pipeline 'base_modelling'
generated using Kedro 1.3.1
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def split_data_base(outlier_handled_data, split):
    target_col = "Outcome"
    X = outlier_handled_data.drop(columns=[target_col])
    y = outlier_handled_data[target_col]
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=split["test"], random_state=split["random_state"]
    )
    return X_train, X_test, y_train.to_frame(), y_test.to_frame()


def train_base_models(base_X_train, base_y_train):
    y = base_y_train.iloc[:, 0]
    models = {
        "RandomForest": RandomForestClassifier(random_state=17),
        "XGBoost": XGBClassifier(random_state=17, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(random_state=17, verbose=-1),
    }
    for model in models.values():
        model.fit(base_X_train, y)
    return models


def evaluate_base_models(base_trained_models, base_X_test, base_y_test):
    y_true = base_y_test.iloc[:, 0]
    results = []
    for name, model in base_trained_models.items():
        y_pred = model.predict(base_X_test)
        y_prob = model.predict_proba(base_X_test)[:, 1]
        results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_true, y_pred), 4),
                "Recall": round(recall_score(y_true, y_pred), 4),
                "Precision": round(precision_score(y_true, y_pred), 4),
                "F1": round(f1_score(y_true, y_pred), 4),
                "AUC": round(roc_auc_score(y_true, y_prob), 4),
            }
        )
    return pd.DataFrame(results).set_index("Model")


def plot_base_model_metrics(base_model_metrics: pd.DataFrame) -> None:
    logger.info(base_model_metrics.to_string())

    metrics = ["Accuracy", "Recall", "Precision", "F1", "AUC"]
    df_plot = base_model_metrics.reset_index()
    df_melted = df_plot.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model")
    plt.title("Comparacao dos Modelos Base")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show(block=True)
