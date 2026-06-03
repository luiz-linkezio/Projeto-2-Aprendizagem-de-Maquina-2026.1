"""Extrai resumo dos resultados para facilitar a criação do relatório."""
import pandas as pd
import numpy as np
import openml

df = pd.read_csv("results/results.csv")
test = df[df["split"] == "test"]

MODELS = ["mambular", "lightgbm", "catboost", "xgboost", "autogluon_default", "autogluon_extreme"]
METRICS = ["auc_ovo", "acc", "g_mean", "ce"]

print("=" * 80)
print("RESUMO DOS RESULTADOS - PROJETO FINAL AM 2026-1")
print("=" * 80)

# ── 1. Tabela de datasets ──────────────────────────────────────────────────────
from experiments.config import DATASETS
print("\n── DATASETS UTILIZADOS ──")
print(f"{'Dataset':<45} {'task_id':>8} {'n':>7} {'regime':<8} {'tipo':<12}")
print("-" * 85)
for d in DATASETS:
    print(f"{d['name']:<45} {d['task_id']:>8} {d['n']:>7} {d['regime']:<8} {d['tipo']:<12}")

# ── 2. Completude ──────────────────────────────────────────────────────────────
print("\n── COMPLETUDE POR DATASET ──")
ok = test[(test["error"].fillna("") == "") & test["auc_ovo"].notna()]
err = test[test["error"].fillna("") != ""]

for d in DATASETS:
    tid = d["task_id"]
    done = ok[ok["task_id"] == tid]["model"].tolist()
    n_done = len(done)
    missing = [m for m in MODELS if m not in done]
    status = "OK" if n_done == 6 else f"{n_done}/6 (falta: {missing})"
    print(f"  {d['name'][:43]:<43}  {status}")

# ── 3. Média de AUC por modelo e regime ───────────────────────────────────────
print("\n── AUC-OVO MÉDIO POR MODELO E REGIME ──")
ds_meta = pd.DataFrame(DATASETS)[["task_id","regime","tipo"]]
merged = ok.merge(ds_meta, on="task_id", suffixes=("", "_meta"))
regime_col = "regime_meta" if "regime_meta" in merged.columns else "regime"
tipo_col   = "tipo_meta"   if "tipo_meta"   in merged.columns else "tipo"
pivot_regime = merged.groupby([regime_col,"model"])["auc_ovo"].mean().unstack("model")
print(pivot_regime.round(4).to_string())

# ── 4. Ranking geral ──────────────────────────────────────────────────────────
print("\n── RANKING GERAL (AUC-OVO MÉDIO - TODOS OS DATASETS) ──")
ranking = ok.groupby("model")["auc_ovo"].mean().sort_values(ascending=False)
for rank, (model, val) in enumerate(ranking.items(), 1):
    count = ok[ok["model"] == model]["auc_ovo"].count()
    print(f"  {rank}. {model:<25} {val:.4f}  (n={count} datasets)")

# ── 5. Tabela completa de métricas por dataset/modelo ─────────────────────────
print("\n── TABELA COMPLETA (TEST) - AUC | ACC | G-Mean | CE | Tempo(s) ──")
for d in DATASETS:
    tid = d["task_id"]
    ds_rows = ok[ok["task_id"] == tid].set_index("model")
    print(f"\n{d['name']} ({d['regime']}, {d['tipo']}, n={d['n']})")
    print(f"  {'Modelo':<22} {'AUC':>7} {'ACC':>7} {'G-Mean':>7} {'CE':>7} {'Tempo(s)':>10}")
    for m in MODELS:
        if m in ds_rows.index:
            r = ds_rows.loc[m]
            auc = f"{r['auc_ovo']:.4f}" if pd.notna(r.get('auc_ovo')) else "  N/A "
            acc = f"{r['acc']:.4f}" if pd.notna(r.get('acc')) else "  N/A "
            gm  = f"{r['g_mean']:.4f}" if pd.notna(r.get('g_mean')) else "  N/A "
            ce  = f"{r['ce']:.4f}" if pd.notna(r.get('ce')) else "  N/A "
            t   = f"{r['time_seconds']:.1f}"
            print(f"  {m:<22} {auc:>7} {acc:>7} {gm:>7} {ce:>7} {t:>10}")
        else:
            print(f"  {m:<22} {'FALHOU':>7}")

# ── 6. Análise binário vs multiclasse ─────────────────────────────────────────
print("\n── AUC-OVO MÉDIO POR TIPO (binário vs multiclasse) ──")
pivot_tipo = merged.groupby([tipo_col,"model"])["auc_ovo"].mean().unstack("model")
print(pivot_tipo.round(4).to_string())

# ── 7. Tempo médio por modelo ─────────────────────────────────────────────────
print("\n── TEMPO MÉDIO POR MODELO (segundos) ──")
tempo = ok.groupby("model")["time_seconds"].mean().sort_values()
for m, t in tempo.items():
    print(f"  {m:<25} {t:>10.1f}s")

print("\nScript concluído.")
