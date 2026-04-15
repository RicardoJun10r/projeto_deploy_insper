"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import (
    show_data,
    clean_data,
    check_missing_outliers_values,
    handle_missing_values,
    handle_outliers,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=show_data,
                inputs="raw_diabetes_data",
                outputs=None,
                name="show_data",
            ),
            Node(
                func=clean_data,
                inputs=["raw_diabetes_data", "params:columns"],
                outputs="cleaned_diabetes_data",
                name="clean_data",
            ),
            Node(
                func=check_missing_outliers_values,
                inputs="cleaned_diabetes_data",
                outputs=None,
                name="check_missing_outliers_values",
            ),
            Node(
                func=handle_missing_values,
                inputs=["cleaned_diabetes_data", "params:missing_values"],
                outputs="missing_handled_data",
                name="missing_values",
            ),
            Node(
                func=handle_outliers,
                inputs=["missing_handled_data", "params:outliers"],
                outputs="outlier_handled_data",
                name="outliers",
            ),
        ]
    )
