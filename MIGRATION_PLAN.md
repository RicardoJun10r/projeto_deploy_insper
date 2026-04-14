# Plano: Notebook → Pipeline Kedro

## Visão Geral

O notebook `notebooks/diabetes_prediction.ipynb` contém um pipeline ML completo para classificação de diabetes. O objetivo é migrar cada etapa para 5 pipelines Kedro organizados.

```
raw_diabetes_data
    │
    ▼ [data_engineering]  ← já existe, precisa expandir
    cleaned_diabetes_data
    missing_handled_data
    outlier_handled_data
         │
    ┌────┴────────────────────┐
    │                         │
    ▼ [base_modelling]        ▼ [feature_engineering]
    baseline metrics          model_input_data
                                   │
                              [data_science]
                              model_metrics
                                   │
                         [hyperparameter_tuning]  ← rodar sob demanda
                         tuned_model_metrics
```

---

## Passo 1 — Expandir `data_engineering`

**Arquivo:** `src/diabetes/pipelines/data_engineering/nodes.py`

Adicione os imports no topo:
```python
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
```

Adicione duas funções novas após `clean_data`:

```python
def handle_missing_values(cleaned_diabetes_data, missing_values):
    df = cleaned_diabetes_data.copy()
    zero_columns = missing_values["zero_columns"]
    n_neighbors = missing_values["knn_neighbors"]

    for col in zero_columns:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    target_col = "Outcome"
    feature_cols = [c for c in df.columns if c != target_col]

    scaler = RobustScaler()
    df_scaled = df[feature_cols].copy()
    df_scaled[:] = scaler.fit_transform(df_scaled)

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = imputer.fit_transform(df_scaled)
    df[feature_cols] = scaler.inverse_transform(df_imputed)

    return df


def handle_outliers(missing_handled_data, outliers):
    df = missing_handled_data.copy()
    q1_pct = outliers["q1"]
    q3_pct = outliers["q3"]
    target_col = "Outcome"
    numerical_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    for col in numerical_cols:
        low = df[col].quantile(q1_pct)
        high = df[col].quantile(q3_pct)
        df[col] = df[col].clip(lower=low, upper=high)

    return df
```

**Arquivo:** `src/diabetes/pipelines/data_engineering/pipeline.py`

```python
from kedro.pipeline import Node, Pipeline
from .nodes import clean_data, handle_missing_values, handle_outliers

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(func=clean_data,
             inputs=["raw_diabetes_data", "params:columns"],
             outputs="cleaned_diabetes_data",
             name="clean_data"),
        Node(func=handle_missing_values,
             inputs=["cleaned_diabetes_data", "params:missing_values"],
             outputs="missing_handled_data",
             name="handle_missing_values"),
        Node(func=handle_outliers,
             inputs=["missing_handled_data", "params:outliers"],
             outputs="outlier_handled_data",
             name="handle_outliers"),
    ])
```

---

## Passo 2 — Criar pipeline `feature_engineering`

Crie a pasta `src/diabetes/pipelines/feature_engineering/` com 3 arquivos:

**`__init__.py`**:
```python
from .pipeline import create_pipeline
__all__ = ["create_pipeline"]
```

**`nodes.py`**:
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler


def create_features(outlier_handled_data):
    df = outlier_handled_data.copy()

    df["NEW_AGE_CAT"] = df["Age"].apply(
        lambda x: "senior" if x >= 50 else ("mature" if x >= 21 else "young")
    )
    df["NEW_BMI"] = pd.cut(df["BMI"],
        bins=[0, 18.5, 24.9, 29.9, np.inf],
        labels=["Underweight", "Healthy", "Overweight", "Obese"])
    df["NEW_GLUCOSE"] = pd.cut(df["Glucose"],
        bins=[0, 140, 200, np.inf],
        labels=["Normal", "Prediabetes", "Diabetes"])
    df["NEW_AGE_BMI_NOM"] = df["NEW_AGE_CAT"].astype(str) + "_" + df["NEW_BMI"].astype(str)
    df["NEW_AGE_GLUCOSE_NOM"] = df["NEW_AGE_CAT"].astype(str) + "_" + df["NEW_GLUCOSE"].astype(str)
    df["NEW_INSULIN_SCORE"] = df["Insulin"].apply(
        lambda x: "Abnormal" if (x < 16 or x > 166) else "Normal"
    )
    df["NEW_GLUCOSE_INSULIN"] = df["Glucose"] * df["Insulin"]
    df.columns = [c.upper() for c in df.columns]
    return df


def encode_features(feature_engineered_data):
    df = feature_engineered_data.copy()
    target_col = "OUTCOME"

    binary_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                   if c != target_col and df[c].nunique() == 2]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    ohe_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if c != target_col and c not in binary_cols]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def scale_features(encoded_data, columns):
    df = encoded_data.copy()
    target_col = "OUTCOME"
    numerical_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    scaler = RobustScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler
```

**`pipeline.py`**:
```python
from kedro.pipeline import Node, Pipeline
from .nodes import create_features, encode_features, scale_features

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(func=create_features,
             inputs="outlier_handled_data",
             outputs="feature_engineered_data",
             name="create_features"),
        Node(func=encode_features,
             inputs="feature_engineered_data",
             outputs="encoded_data",
             name="encode_features"),
        Node(func=scale_features,
             inputs=["encoded_data", "params:columns"],
             outputs=["model_input_data", "feature_scaler"],
             name="scale_features"),
    ])
```

---

## Passo 3 — Criar pipeline `base_modelling`

Crie a pasta `src/diabetes/pipelines/base_modelling/` com 3 arquivos:

**`__init__.py`**:
```python
from .pipeline import create_pipeline
__all__ = ["create_pipeline"]
```

**`nodes.py`**:
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def split_data_base(outlier_handled_data, split):
    target_col = "Outcome"
    X = outlier_handled_data.drop(columns=[target_col])
    y = outlier_handled_data[target_col]
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=split["test_size"], random_state=split["random_state"]
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
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "F1": round(f1_score(y_true, y_pred), 4),
            "AUC": round(roc_auc_score(y_true, y_prob), 4),
        })
    return pd.DataFrame(results).set_index("Model")
```

**`pipeline.py`**:
```python
from kedro.pipeline import Node, Pipeline
from .nodes import split_data_base, train_base_models, evaluate_base_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(func=split_data_base,
             inputs=["outlier_handled_data", "params:split"],
             outputs=["base_X_train", "base_X_test", "base_y_train", "base_y_test"],
             name="split_data_base"),
        Node(func=train_base_models,
             inputs=["base_X_train", "base_y_train"],
             outputs="base_trained_models",
             name="train_base_models"),
        Node(func=evaluate_base_models,
             inputs=["base_trained_models", "base_X_test", "base_y_test"],
             outputs="base_model_metrics",
             name="evaluate_base_models"),
    ])
```

---

## Passo 4 — Criar pipeline `data_science`

Crie a pasta `src/diabetes/pipelines/data_science/` com 3 arquivos:

**`__init__.py`**:
```python
from .pipeline import create_pipeline
__all__ = ["create_pipeline"]
```

**`nodes.py`**:
> Atenção: os dados já chegam escalados do `feature_engineering`, então **não** aplique RobustScaler aqui. A coluna target agora é `OUTCOME` (maiúscula).

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def split_data(model_input_data, split):
    target_col = "OUTCOME"
    X = model_input_data.drop(columns=[target_col])
    y = model_input_data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split["test_size"], random_state=split["random_state"]
    )
    return X_train, X_test, y_train.to_frame(), y_test.to_frame()


def train_models(X_train, y_train):
    y = y_train.iloc[:, 0]
    models = {
        "RandomForest": RandomForestClassifier(random_state=17),
        "XGBoost": XGBClassifier(random_state=17, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(random_state=17, verbose=-1),
    }
    for model in models.values():
        model.fit(X_train, y)
    return models


def evaluate_models(trained_models, X_test, y_test):
    y_true = y_test.iloc[:, 0]
    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "F1": round(f1_score(y_true, y_pred), 4),
            "AUC": round(roc_auc_score(y_true, y_prob), 4),
        })
    return pd.DataFrame(results).set_index("Model")
```

**`pipeline.py`**:
```python
from kedro.pipeline import Node, Pipeline
from .nodes import split_data, train_models, evaluate_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(func=split_data,
             inputs=["model_input_data", "params:split"],
             outputs=["X_train", "X_test", "y_train", "y_test"],
             name="split_data"),
        Node(func=train_models,
             inputs=["X_train", "y_train"],
             outputs="trained_models",
             name="train_models"),
        Node(func=evaluate_models,
             inputs=["trained_models", "X_test", "y_test"],
             outputs="model_metrics",
             name="evaluate_models"),
    ])
```

---

## Passo 5 — Criar pipeline `hyperparameter_tuning`

Crie a pasta `src/diabetes/pipelines/hyperparameter_tuning/` com 3 arquivos:

**`__init__.py`**:
```python
from .pipeline import create_pipeline
__all__ = ["create_pipeline"]
```

**`nodes.py`**:
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def tune_random_forest(X_train, y_train, rf_param_grid, cv_folds):
    y = y_train.iloc[:, 0]
    gs = GridSearchCV(RandomForestClassifier(random_state=17),
                      rf_param_grid, cv=cv_folds, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y)
    return gs.best_estimator_


def tune_xgboost(X_train, y_train, xgb_param_grid, cv_folds):
    y = y_train.iloc[:, 0]
    gs = GridSearchCV(XGBClassifier(random_state=17, eval_metric="logloss", verbosity=0),
                      xgb_param_grid, cv=cv_folds, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y)
    return gs.best_estimator_


def tune_lightgbm(X_train, y_train, lgbm_param_grid, cv_folds):
    y = y_train.iloc[:, 0]
    gs = GridSearchCV(LGBMClassifier(random_state=17, verbose=-1),
                      lgbm_param_grid, cv=cv_folds, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y)
    return gs.best_estimator_


def evaluate_tuned_models(tuned_rf_model, tuned_xgb_model, tuned_lgbm_model, X_test, y_test):
    y_true = y_test.iloc[:, 0]
    models = {
        "RF_Tuned": tuned_rf_model,
        "XGB_Tuned": tuned_xgb_model,
        "LGBM_Tuned": tuned_lgbm_model,
    }
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "F1": round(f1_score(y_true, y_pred), 4),
            "AUC": round(roc_auc_score(y_true, y_prob), 4),
        })
    return pd.DataFrame(results).set_index("Model")
```

**`pipeline.py`**:
> Os 3 nodes de tuning não dependem entre si — o Kedro pode rodá-los em paralelo.

```python
from kedro.pipeline import Node, Pipeline
from .nodes import tune_random_forest, tune_xgboost, tune_lightgbm, evaluate_tuned_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(func=tune_random_forest,
             inputs=["X_train", "y_train", "params:rf_param_grid", "params:cv_folds"],
             outputs="tuned_rf_model",
             name="tune_rf"),
        Node(func=tune_xgboost,
             inputs=["X_train", "y_train", "params:xgb_param_grid", "params:cv_folds"],
             outputs="tuned_xgb_model",
             name="tune_xgb"),
        Node(func=tune_lightgbm,
             inputs=["X_train", "y_train", "params:lgbm_param_grid", "params:cv_folds"],
             outputs="tuned_lgbm_model",
             name="tune_lgbm"),
        Node(func=evaluate_tuned_models,
             inputs=["tuned_rf_model", "tuned_xgb_model", "tuned_lgbm_model", "X_test", "y_test"],
             outputs="tuned_model_metrics",
             name="evaluate_tuned"),
    ])
```

---

## Passo 6 — Atualizar `conf/base/catalog.yml`

Acrescente ao final do arquivo:

```yaml
# data_engineering
missing_handled_data:
  type: pandas.CSVDataset
  filepath: data/02_intermediate/missing-handled.csv
  metadata: {kedro-viz: {layer: intermediate}}

outlier_handled_data:
  type: pandas.CSVDataset
  filepath: data/03_primary/outlier-handled.csv
  metadata: {kedro-viz: {layer: primary}}

# feature_engineering
feature_engineered_data:
  type: pandas.CSVDataset
  filepath: data/04_feature/feature-engineered.csv
  metadata: {kedro-viz: {layer: feature}}

encoded_data:
  type: pandas.CSVDataset
  filepath: data/04_feature/encoded.csv
  metadata: {kedro-viz: {layer: feature}}

model_input_data:
  type: pandas.CSVDataset
  filepath: data/05_model_input/model-input.csv
  metadata: {kedro-viz: {layer: model_input}}

feature_scaler:
  type: pickle.PickleDataset
  filepath: data/06_models/feature_scaler.pkl

# base_modelling
base_X_train:
  type: pandas.CSVDataset
  filepath: data/05_model_input/base_X_train.csv
base_X_test:
  type: pandas.CSVDataset
  filepath: data/05_model_input/base_X_test.csv
base_y_train:
  type: pandas.CSVDataset
  filepath: data/05_model_input/base_y_train.csv
base_y_test:
  type: pandas.CSVDataset
  filepath: data/05_model_input/base_y_test.csv
base_trained_models:
  type: pickle.PickleDataset
  filepath: data/06_models/base_trained_models.pkl
base_model_metrics:
  type: pandas.CSVDataset
  filepath: data/08_reporting/base_model_metrics.csv

# data_science
X_train:
  type: pandas.CSVDataset
  filepath: data/05_model_input/X_train.csv
X_test:
  type: pandas.CSVDataset
  filepath: data/05_model_input/X_test.csv
y_train:
  type: pandas.CSVDataset
  filepath: data/05_model_input/y_train.csv
y_test:
  type: pandas.CSVDataset
  filepath: data/05_model_input/y_test.csv
trained_models:
  type: pickle.PickleDataset
  filepath: data/06_models/trained_models.pkl
model_metrics:
  type: pandas.CSVDataset
  filepath: data/08_reporting/model_metrics.csv

# hyperparameter_tuning
tuned_rf_model:
  type: pickle.PickleDataset
  filepath: data/06_models/tuned_rf_model.pkl
tuned_xgb_model:
  type: pickle.PickleDataset
  filepath: data/06_models/tuned_xgb_model.pkl
tuned_lgbm_model:
  type: pickle.PickleDataset
  filepath: data/06_models/tuned_lgbm_model.pkl
tuned_model_metrics:
  type: pandas.CSVDataset
  filepath: data/08_reporting/tuned_model_metrics.csv
```

---

## Passo 7 — Atualizar `conf/base/parameters.yml`

Acrescente ao final do arquivo:

```yaml
missing_values:
  zero_columns: [Glucose, BloodPressure, SkinThickness, Insulin, BMI]
  knn_neighbors: 5

outliers:
  q1: 0.05
  q3: 0.95

split:
  test_size: 0.30
  random_state: 17

rf_param_grid:
  n_estimators: [100, 200, 300]
  max_depth: [null, 5, 10]
  min_samples_split: [2, 5, 10]

xgb_param_grid:
  n_estimators: [50, 100, 200]
  learning_rate: [0.1, 0.5, 1.0]

lgbm_param_grid:
  n_estimators: [50, 100, 200]
  learning_rate: [0.1, 0.5, 1.0]

cv_folds: 5
```

---

## Como testar cada passo

```bash
kedro run --pipeline data_engineering
kedro run --pipeline feature_engineering
kedro run --pipeline base_modelling
kedro run --pipeline data_science
kedro run --pipeline hyperparameter_tuning   # mais lento, rodar por último
```

Os outputs finais ficam em:

| Arquivo | Conteúdo |
|---|---|
| `data/08_reporting/base_model_metrics.csv` | Métricas baseline (sem feature engineering) |
| `data/08_reporting/model_metrics.csv` | Métricas com feature engineering |
| `data/08_reporting/tuned_model_metrics.csv` | Métricas após hyperparameter tuning |
| `data/06_models/*.pkl` | Modelos serializados |
