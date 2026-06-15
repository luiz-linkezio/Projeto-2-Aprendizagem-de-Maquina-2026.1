from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Pré-processamento para modelos sklearn ────────────────────────────────────

def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    transformers: list = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            cat_cols,
        ))
    return ColumnTransformer(transformers, remainder="passthrough")


def _auc_scorer(estimator, X, y) -> float:
    y_proba = estimator.predict_proba(X)
    n_classes = len(np.unique(y))
    if n_classes == 2:
        return float(roc_auc_score(y, y_proba[:, 1]))
    return float(roc_auc_score(y, y_proba, multi_class="ovo", average="macro"))


def _sklearn_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int, seed: int) -> float:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=_auc_scorer, n_jobs=1, error_score="raise")
    return float(scores.mean())


# ── CV personalizado para Mambular (max_epochs vai no fit, não no construtor) ─

def _cuda_cleanup() -> None:
    try:
        import gc
        import os
        import shutil
        import torch
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        # Remove checkpoints do Lightning para libertar referências de memória
        if os.path.exists("model_checkpoints"):
            shutil.rmtree("model_checkpoints", ignore_errors=True)
    except Exception:
        pass


def _sanitize_for_mambular(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object or str(df[col].dtype) == "category":
            df[col] = df[col].astype(str).replace("nan", "missing").replace("None", "missing")
    return df


def _mambular_cv(params: dict, X: pd.DataFrame, y: pd.Series, cv_folds: int, seed: int) -> float:
    from deeptab.models import MambularClassifier

    X = _sanitize_for_mambular(X)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = None
        try:
            model = MambularClassifier(**params)
            model.fit(X_tr, y_tr, max_epochs=20, batch_size=32)
            y_proba = model.predict_proba(X_val)
            n_classes = len(np.unique(y_tr))
            if n_classes == 2:
                proba_col = y_proba[:, 0] if y_proba.shape[1] == 1 else y_proba[:, 1]
                scores.append(roc_auc_score(y_val, proba_col))
            else:
                scores.append(roc_auc_score(y_val, y_proba, multi_class="ovo", average="macro"))
        finally:
            # Sempre limpa VRAM — mesmo em caso de OOM ou outra exceção
            if model is not None:
                del model
            _cuda_cleanup()
    return float(np.mean(scores)) if scores else 0.0


# ── Funções de tuning ─────────────────────────────────────────────────────────

def tune_mambular(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int, cv_folds: int, seed: int) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "d_model": trial.suggest_int("d_model", 16, 32, step=16),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "numerical_preprocessing": trial.suggest_categorical(
                "numerical_preprocessing", ["standardization", "quantile"]
            ),
        }
        return _mambular_cv(params, X_train, y_train, cv_folds, seed)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int, cv_folds: int, seed: int) -> dict:
    from lightgbm import LGBMClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": seed, "verbose": -1, "n_jobs": -1,
        }
        pipeline = Pipeline([("prep", _make_preprocessor(X_train)), ("model", LGBMClassifier(**params))])
        return _sklearn_cv(pipeline, X_train, y_train, cv_folds, seed)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_catboost(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int, cv_folds: int, seed: int) -> dict:
    from catboost import CatBoostClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_seed": seed, "verbose": 0,
        }
        pipeline = Pipeline([("prep", _make_preprocessor(X_train)), ("model", CatBoostClassifier(**params))])
        return _sklearn_cv(pipeline, X_train, y_train, cv_folds, seed)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int, cv_folds: int, seed: int) -> dict:
    from xgboost import XGBClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": seed, "eval_metric": "logloss", "n_jobs": -1,
        }
        pipeline = Pipeline([("prep", _make_preprocessor(X_train)), ("model", XGBClassifier(**params))])
        return _sklearn_cv(pipeline, X_train, y_train, cv_folds, seed)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


_TUNE_FNS = {
    "mambular": tune_mambular,
    "lightgbm": tune_lightgbm,
    "catboost": tune_catboost,
    "xgboost": tune_xgboost,
}


def tune_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
               n_trials: int, cv_folds: int, seed: int) -> dict:
    return _TUNE_FNS[model_name](X_train, y_train, n_trials, cv_folds, seed)


# ── Construção do modelo final com os melhores parâmetros ─────────────────────

def build_model(model_name: str, best_params: dict, X_train: pd.DataFrame):
    if model_name == "mambular":
        from deeptab.models import MambularClassifier
        return MambularClassifier(**best_params)

    preprocessor = _make_preprocessor(X_train)

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return Pipeline([("prep", preprocessor), ("model", LGBMClassifier(**best_params))])
    if model_name == "catboost":
        from catboost import CatBoostClassifier
        return Pipeline([("prep", preprocessor), ("model", CatBoostClassifier(**best_params))])
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return Pipeline([("prep", preprocessor), ("model", XGBClassifier(**best_params))])

    raise ValueError(f"Modelo desconhecido: {model_name}")
