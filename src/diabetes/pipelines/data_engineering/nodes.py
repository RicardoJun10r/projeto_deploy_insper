"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.3.1
"""

import logging
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)

def clean_data(
    raw_diabetes_data: pd.DataFrame,
    columns: dict[str, Any]
) -> pd.DataFrame:
    raw_df = raw_diabetes_data.copy()
    
    all_columns = [
        item
        for v in columns.values()
        for item in (v if isinstance(v, list) else [v])
        if item in raw_df.columns
    ]
    
    cleaned_df = raw_df[all_columns].copy()
    
    for c in columns['numerical']:
        cleaned_df[c] = pd.to_numeric(cleaned_df[c], errors='coerce').fillna(0)
    
    logger.info("Dados limpos: %d linhas, %d colunas", len(cleaned_df), len(cleaned_df.columns))
    return cleaned_df