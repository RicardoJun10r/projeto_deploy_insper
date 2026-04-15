"""
This is a boilerplate pipeline 'base_modelling'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import (
    split_data_base,
    train_base_models,
    evaluate_base_models,
    plot_base_model_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split_data_base,
                inputs=["outlier_handled_data", "params:split"],
                outputs=["base_X_train", "base_X_test", "base_y_train", "base_y_test"],
                name="split_data_base",
            ),
            Node(
                func=train_base_models,
                inputs=[
                    "base_X_train",
                    "base_y_train",
                    "params:models",
                    "params:model_params",
                ],
                outputs="base_trained_models",
                name="train_base_models",
            ),
            Node(
                func=evaluate_base_models,
                inputs=[
                    "base_trained_models",
                    "base_X_test",
                    "base_y_test",
                    "params:models",
                ],
                outputs="base_model_metrics",
                name="evaluate_base_models",
            ),
            Node(
                func=plot_base_model_metrics,
                inputs="base_model_metrics",
                outputs=None,
                name="plot_base_model_metrics",
            ),
        ]
    )
