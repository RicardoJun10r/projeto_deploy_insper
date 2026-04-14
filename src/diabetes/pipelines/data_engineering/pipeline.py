"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import show_data, detecting_categorical_numerical_variables, clean_data


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
                func=detecting_categorical_numerical_variables,
                inputs="raw_diabetes_data",
                outputs=None,
                name="detect_cat_num_var",
            ),
            Node(
                func=clean_data,
                inputs=["raw_diabetes_data", "params:columns"],
                outputs="cleaned_diabetes_data",
                name="clean_data",
            ),
        ]
    )
