"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import (clean_data)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=clean_data,
            inputs=["raw_diabetes_data", "params:columns"],
            outputs="cleaned_diabetes_data",
            name="clean_data"
        )
    ])
