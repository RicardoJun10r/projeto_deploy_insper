"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import create_features, encode_features, scale_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=create_features,
                inputs="outlier_handled_data",
                outputs="feature_engineered_data",
                name="create_features",
            ),
            Node(
                func=encode_features,
                inputs="feature_engineered_data",
                outputs="encoded_data",
                name="encode_features",
            ),
            Node(
                func=scale_features,
                inputs="encoded_data",
                outputs=["model_input_data", "feature_scaler"],
                name="scale_features",
            ),
        ]
    )
