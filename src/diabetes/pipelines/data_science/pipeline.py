"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import evaluate_models, plot_model_metrics, split_data, train_models


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split_data,
                inputs=["model_input_data", "params:split"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            Node(
                func=train_models,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:models",
                    "params:model_params",
                ],
                outputs="trained_models",
                name="train_models",
            ),
            Node(
                func=evaluate_models,
                inputs=["trained_models", "X_test", "y_test", "params:models"],
                outputs="model_metrics",
                name="evaluate_models",
            ),
            Node(
                func=plot_model_metrics,
                inputs="model_metrics",
                outputs=None,
                name="plot_model_metrics",
            ),
        ]
    )
