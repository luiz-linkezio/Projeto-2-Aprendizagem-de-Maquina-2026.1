"""Ponto de entrada do experimento. Executar com:  python -m experiments.runner"""
from __future__ import annotations

import csv
import gc
import json
import logging
import os
import shutil
import time

import numpy as np
import pandas as pd

# Permite que o PyTorch aloque segmentos expansíveis — reduz fragmentação de VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# Desabilita o OOM killer do Ray para evitar loop infinito de retries
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")
# Limita o object store do Ray a 3 GB
os.environ.setdefault("RAY_object_store_memory", str(3 * 1024 ** 3))

from experiments.config import (
    AUTOGLUON_TIME_DEFAULT,
    AUTOGLUON_TIME_EXTREME,
    DATASETS,
    MODELS,
    OPTUNA_CV_FOLDS,
    OPTUNA_N_TRIALS,
    SEED,
    TEST_SIZE,
)
from experiments.data_loader import load_dataset
from experiments.metrics import compute_metrics
from experiments.models import build_model, tune_model

RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
BEST_PARAMS_JSON = os.path.join(RESULTS_DIR, "best_params.json")
LOG_FILE = os.path.join(RESULTS_DIR, "runner.log")

_CSV_COLS = ["task_id", "dataset_name", "regime", "tipo", "model",
             "split", "auc_ovo", "acc", "g_mean", "ce", "time_seconds", "error"]


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logger() -> logging.Logger:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger = logging.getLogger("runner")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _mem_info() -> str:
    try:
        import psutil
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return (f"RAM {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB "
                f"({vm.percent:.0f}%)  SWAP {sw.used/1e9:.1f}/{sw.total/1e9:.1f} GB")
    except Exception:
        return ""


def _vram_info() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return f"VRAM {used/1e9:.1f}/{total/1e9:.1f} GB"
    except Exception:
        return ""


def _ram_available_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 999.0


def _vram_cleanup() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# RAM mínima exigida antes de iniciar cada modelo (em GB)
_RAM_GUARD = {"autogluon_extreme": 6.0, "autogluon_default": 4.0, "default": 2.0}


# ── Helpers de I/O ────────────────────────────────────────────────────────────

def _computed_pairs() -> set[tuple]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    ok = df[(df["split"] == "test") & (df["error"].fillna("") == "") & df["auc_ovo"].notna()]
    done = set(zip(ok["task_id"], ok["model"]))
    # Pares com 3+ erros são tratados como definitivamente falhos — não retenta mais
    errors = df[(df["split"] == "test") & (df["error"].fillna("") != "")]
    too_many = errors.groupby(["task_id", "model"]).size()
    for (tid, m), count in too_many.items():
        if count >= 3:
            done.add((tid, m))
    return done


def _append_row(row: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_params(task_id: int, model_name: str, params: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data: dict = {}
    if os.path.exists(BEST_PARAMS_JSON):
        with open(BEST_PARAMS_JSON) as f:
            data = json.load(f)
    data.setdefault(str(task_id), {})[model_name] = params
    with open(BEST_PARAMS_JSON, "w") as f:
        json.dump(data, f, indent=2)


def _make_row(info: dict, model: str, split: str, metrics: dict,
              elapsed: float, error: str = "") -> dict:
    return {
        "task_id": info["task_id"], "dataset_name": info["name"],
        "regime": info["regime"], "tipo": info["tipo"],
        "model": model, "split": split,
        **{k: metrics.get(k) for k in ("auc_ovo", "acc", "g_mean", "ce")},
        "time_seconds": round(elapsed, 2), "error": error,
    }


# ── AutoGluon ─────────────────────────────────────────────────────────────────

def _run_autogluon(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame,
    preset: str, time_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from autogluon.tabular import TabularDataset, TabularPredictor
    # Esconde GPU para AutoGluon/Ray — evita falhas de XGBoost-CUDA nos workers
    _old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    n_classes = len(y_train.unique())
    eval_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"

    train_df = X_train.copy()
    train_df["__y__"] = y_train.values
    train_data = TabularDataset(train_df)

    predictor = TabularPredictor(label="__y__", eval_metric=eval_metric, verbosity=0)
    predictor.fit(
        train_data,
        presets=preset,
        time_limit=time_limit,
        num_cpus=2,
        num_gpus=0,
        num_bag_folds=3,
        num_stack_levels=0,
        # Desativa Ray — executa modelos sequencialmente, sem fork de workers
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )

    def _extract(X: pd.DataFrame):
        proba_df = predictor.predict_proba(X)
        proba_df = proba_df[sorted(proba_df.columns)]
        proba = proba_df.values
        pred = proba.argmax(axis=1)
        return pred, proba

    y_pred_test,  y_proba_test  = _extract(X_test)
    y_pred_train, y_proba_train = _extract(X_train)

    del predictor
    gc.collect()
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    if os.path.exists("AutogluonModels"):
        shutil.rmtree("AutogluonModels", ignore_errors=True)

    if _old_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _old_cuda

    return y_pred_test, y_proba_test, y_pred_train, y_proba_train


# ── Loop principal ────────────────────────────────────────────────────────────

def run() -> None:
    log = _setup_logger()
    computed = _computed_pairs()
    log.info(f"Iniciando runner  |  pares já computados: {len(computed)}  |  {_mem_info()}")

    for info in DATASETS:
        task_id = info["task_id"]
        name = info["name"]
        log.info(f"{'='*55}")
        log.info(f"DATASET: {name}  (task_id={task_id}, regime={info['regime']}, n={info['n']})")

        # Limpa VRAM residual de runs anteriores antes de cada dataset
        gc.collect()
        _vram_cleanup()

        try:
            X_train, X_test, y_train, y_test = load_dataset(task_id, SEED, TEST_SIZE)[:4]
            log.debug(f"  Carregado: train={len(X_train)}  test={len(X_test)}  feats={X_train.shape[1]}")
        except Exception as exc:
            log.error(f"  ERRO ao carregar dataset: {exc}", exc_info=True)
            continue

        for model_name in MODELS:
            if (task_id, model_name) in computed:
                log.debug(f"  [{model_name}] já computado — pulando")
                continue

            # Guard de memória — aborta se RAM disponível for insuficiente
            ram_needed = _RAM_GUARD.get(model_name, _RAM_GUARD["default"])
            ram_free = _ram_available_gb()
            if ram_free < ram_needed:
                log.warning(f"  [{model_name}] PULADO — RAM disponível {ram_free:.1f} GB < {ram_needed:.1f} GB necessários  |  {_mem_info()}")
                empty = {"auc_ovo": None, "acc": None, "g_mean": None, "ce": None}
                _append_row(_make_row(info, model_name, "test", empty, 0.0,
                                     f"RAM insuficiente: {ram_free:.1f} GB disponíveis"))
                computed.add((task_id, model_name))
                continue

            log.info(f"  [{model_name}] iniciando...  |  {_mem_info()}  {_vram_info()}")
            t0 = time.time()
            error = ""

            try:
                if "autogluon" in model_name:
                    preset = "best_quality" if model_name == "autogluon_extreme" else "medium_quality"
                    tl = AUTOGLUON_TIME_EXTREME if model_name == "autogluon_extreme" else AUTOGLUON_TIME_DEFAULT
                    y_pred, y_proba, y_pred_tr, y_proba_tr = _run_autogluon(
                        X_train, y_train, X_test, preset, tl
                    )
                else:
                    best_params = tune_model(model_name, X_train, y_train,
                                             OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, SEED)
                    _save_params(task_id, model_name, best_params)
                    log.debug(f"  [{model_name}] best_params={best_params}")
                    model = build_model(model_name, best_params, X_train)

                    if model_name == "mambular":
                        model.fit(X_train, y_train, max_epochs=50)
                    else:
                        model.fit(X_train, y_train)

                    y_pred    = model.predict(X_test)
                    y_proba   = model.predict_proba(X_test)
                    y_pred_tr = model.predict(X_train)
                    y_proba_tr = model.predict_proba(X_train)

                    if model_name == "mambular":
                        del model
                        gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

                elapsed = time.time() - t0
                test_m  = compute_metrics(y_test.values,  np.array(y_pred),    np.array(y_proba))
                train_m = compute_metrics(y_train.values, np.array(y_pred_tr), np.array(y_proba_tr))

                _append_row(_make_row(info, model_name, "test",  test_m,  elapsed))
                _append_row(_make_row(info, model_name, "train", train_m, elapsed))

                log.info(f"  [{model_name}] OK  AUC={test_m['auc_ovo']:.4f}  "
                         f"ACC={test_m['acc']:.4f}  G-Mean={test_m['g_mean']:.4f}  "
                         f"CE={test_m['ce']:.4f}  t={elapsed:.0f}s")

            except Exception as exc:
                elapsed = time.time() - t0
                error = str(exc)
                log.error(f"  [{model_name}] ERRO após {elapsed:.0f}s: {exc}", exc_info=True)
                empty = {"auc_ovo": None, "acc": None, "g_mean": None, "ce": None}
                _append_row(_make_row(info, model_name, "test", empty, elapsed, error))

            finally:
                if model_name == "mambular":
                    gc.collect()
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            free = torch.cuda.mem_get_info()[0] / 1024**3
                            log.debug(f"  [mambular] VRAM livre após cleanup: {free:.1f} GB")
                    except Exception:
                        pass

            computed.add((task_id, model_name))
            gc.collect()
            log.debug(f"  [{model_name}] pós-GC  |  {_mem_info()}")

    log.info("Runner finalizado.")


if __name__ == "__main__":
    run()
