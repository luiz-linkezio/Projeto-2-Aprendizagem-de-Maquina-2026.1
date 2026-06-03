"""Análise estatística dos resultados. Executar com:  python -m experiments.analysis"""
from __future__ import annotations

import os

import baycomp
import matplotlib.pyplot as plt
import pandas as pd
from autorank import autorank, create_report, plot_stats

RESULTS_CSV = "results/results.csv"
PLOTS_DIR = "results/plots"

METRIC = "auc_ovo"  # métrica principal para os testes estatísticos
ROPE = 0.01         # região de equivalência prática (1 pp)

MODEL_LABELS = {
    "mambular":          "Mambular",
    "lightgbm":          "LightGBM",
    "catboost":          "CatBoost",
    "xgboost":           "XGBoost",
    "autogluon_default": "AG-Default",
    "autogluon_extreme": "AG-Extreme",
}

BASELINES = ["lightgbm", "catboost", "xgboost", "autogluon_default", "autogluon_extreme"]


def _load_pivot(metric: str = METRIC) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    test = df[df["split"] == "test"].dropna(subset=[metric])
    pivot = test.pivot_table(index="dataset_name", columns="model", values=metric)
    pivot.rename(columns=MODEL_LABELS, inplace=True)
    return pivot


def run_classical(pivot: pd.DataFrame) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n── Teste de Friedman + Nemenyi/Holm + Diagrama de Diferença Crítica ──")
    result = autorank(pivot, alpha=0.05, verbose=True)
    create_report(result)

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_stats(result, ax=ax)
    ax.set_title(f"Critical Difference Diagram — {METRIC.upper()}")
    path = os.path.join(PLOTS_DIR, "cd_diagram.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  CD diagram salvo em {path}")


def run_bayesian(pivot: pd.DataFrame) -> None:
    print("\n── Análise Bayesiana (Mambular vs cada baseline) ──")
    rows = []
    mambular_scores = pivot[MODEL_LABELS["mambular"]].values

    for baseline in BASELINES:
        label = MODEL_LABELS[baseline]
        if label not in pivot.columns:
            continue
        baseline_scores = pivot[label].values

        # Bayesian signed-rank test
        p_left, p_rope, p_right = baycomp.two_on_multiple(
            mambular_scores, baseline_scores, rope=ROPE
        )
        rows.append({
            "baseline": label,
            "P(Mambular > baseline)": round(p_left, 4),
            "P(equivalente)":         round(p_rope, 4),
            "P(baseline > Mambular)": round(p_right, 4),
        })
        print(f"  Mambular vs {label:12s} | "
              f"P(M>B)={p_left:.3f}  P(rope)={p_rope:.3f}  P(B>M)={p_right:.3f}")

        # Gráfico Bayesiano por par
        try:
            result_plot = baycomp.two_on_multiple(
                mambular_scores, baseline_scores, rope=ROPE, plot=True,
                names=("Mambular", label)
            )
            fig = result_plot[0] if isinstance(result_plot, tuple) else result_plot
            if fig is not None:
                path = os.path.join(PLOTS_DIR, f"bayesian_mambular_vs_{baseline}.png")
                fig.savefig(path, dpi=150)
                plt.close(fig)
        except Exception:
            pass

    results_df = pd.DataFrame(rows)
    out = os.path.join("results", "bayesian_results.csv")
    results_df.to_csv(out, index=False)
    print(f"\n  Resultados salvos em {out}")


def regime_summary() -> None:
    df = pd.read_csv(RESULTS_CSV)
    test = df[df["split"] == "test"]
    summary = (
        test.groupby(["regime", "model"])[METRIC]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print("\n── Análise por Regime ──")
    print(summary.to_string())
    summary.to_csv(os.path.join("results", "regime_summary.csv"))


def main() -> None:
    if not os.path.exists(RESULTS_CSV):
        print(f"Arquivo de resultados não encontrado: {RESULTS_CSV}")
        print("Execute primeiro:  python -m experiments.runner")
        return

    pivot = _load_pivot()
    print(f"Datasets com resultados completos: {len(pivot)}")
    print(f"Modelos: {list(pivot.columns)}\n")

    run_classical(pivot)
    run_bayesian(pivot)
    regime_summary()


if __name__ == "__main__":
    main()
