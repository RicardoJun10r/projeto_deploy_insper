"""
This is a boilerplate pipeline 'hyperparameter_tuning'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import evaluate_tuned_models, plot_tuned_model_metrics, tune_models


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=tune_models,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:models",
                    "params:model_params",
                    "params:tuning",
                ],
                outputs="tuned_models",
                name="tune_models",
            ),
            Node(
                func=evaluate_tuned_models,
                inputs=["tuned_models", "X_test", "y_test", "params:models"],
                outputs="tuned_model_metrics",
                name="evaluate_tuned_models",
            ),
            Node(
                func=plot_tuned_model_metrics,
                inputs="tuned_model_metrics",
                outputs=None,
                name="plot_tuned_model_metrics",
            ),
        ]
    )
