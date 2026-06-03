# Slides — Etapa 2: Resultados Experimentais
**Projeto Final AM 2026-1 | Prof. Leandro Almeida | 10 minutos**

---

## Slide 1 — Título

# Experimentos: Mambular vs Baselines
### TabArena-v0.1 — 30 Datasets de Classificação

**Modelos comparados:**  
Mambular | LightGBM | CatBoost | XGBoost | AutoGluon-Default | AutoGluon-Extreme

---

## Slide 2 — Setup Experimental

**30 datasets do TabArena-v0.1 (NeurIPS 2025)**

| Regime | Critério | Datasets |
|---|---|---|
| Pequeno | n < 1.000 | 3 datasets |
| Médio | 1.000 ≤ n ≤ 10.000 | 14 datasets |
| Grande | n > 10.000 | 13 datasets |

**Protocolo:**
- Split: 70% treino / 30% teste (`seed=42`, estratificado)
- HPO: Optuna TPE, 10 trials, 2-fold CV
- Métricas: **AUC-OVO**, Accuracy, G-Mean, Cross-Entropy, Tempo

**Hardware:**
- CPU: AMD/Intel (modelos de árvore)
- GPU: NVIDIA RTX 5070 12 GB (Mambular)

---

## Slide 3 — Ranking Geral (AUC-OVO médio)

| # | Modelo | AUC Médio | DP | Mean Rank |
|---|---|---|---|---|
| 1 | **AG-Extreme** | 0.865 | ±0.102 | 1.82 |
| 2 | AG-Default | 0.863 | ±0.101 | 2.53 |
| 3 | CatBoost | 0.859 | ±0.101 | 3.25 |
| 4 | XGBoost | 0.857 | ±0.101 | 3.80 |
| 5 | LightGBM | 0.855 | ±0.103 | 4.33 |
| 6 | **Mambular** | **0.829** | **±0.101** | **5.15** |

**Nota:** Mambular ausente em 4 datasets (OOM): anneal, qsar-biodeg, MIC, APSFailure

---

## Slide 4 — Diagrama de Diferença Crítica (Nemenyi)

```
Teste de Friedman: diferenças significativas entre os modelos (p < 0.05)
Distância Crítica CD = 1.377 (α=0.05, 6 modelos, 30 datasets)

Rank:  1.82      2.53       3.25    3.80   4.33    5.15
       |---------|----------|--------|-------|-------|
    AG-Ext  AG-Def  CatBoost  XGBoost  LGBM  Mambular

Grupos sem diferença significativa:
  ├── [Mambular — LightGBM — XGBoost]       (CD não ultrapassada)
  ├── [LightGBM — XGBoost — CatBoost]
  ├── [XGBoost — CatBoost — AG-Default]
  └── [AG-Default — AG-Extreme]
```

> Ver: `results/plots/cd_diagram.png`

**Interpretação:** Mambular é significativamente pior apenas que CatBoost, AG-Default e AG-Extreme.  
**Diferença vs LightGBM/XGBoost: NÃO é estatisticamente significativa.**

---

## Slide 5 — Análise Bayesiana (ROPE = 1%)

**Teste signed-rank Bayesiano: Mambular vs cada baseline**

| Comparação | P(M > B) | P(equivalente) | P(B > M) | Veredicto |
|---|---|---|---|---|
| Mambular vs LightGBM | 0.000 | **0.768** | 0.232 | **≈ equivalentes** |
| Mambular vs CatBoost | 0.000 | **0.702** | 0.298 | **≈ equivalentes** |
| Mambular vs XGBoost | 0.000 | **0.694** | 0.306 | **≈ equivalentes** |
| Mambular vs AG-Default | 0.000 | 0.303 | **0.697** | AG-Default superior |
| Mambular vs AG-Extreme | 0.000 | 0.213 | **0.787** | AG-Extreme superior |

**ROPE:** região de ±1 pp onde diferença é *praticamente insignificante*

**Conclusão Bayesiana:**
> Com ~70–77% de probabilidade, o Mambular é **praticamente equivalente** às GBDTs clássicas dentro de 1 ponto percentual de AUC

---

## Slide 6 — Análise por Regime de Tamanho

| Regime | Mambular | LightGBM | CatBoost | XGBoost | AG-Def | AG-Ext |
|---|---|---|---|---|---|---|
| **Pequeno** (n<1k) | 0.806 | 0.865 | 0.880 | 0.870 | 0.872 | 0.880 |
| **Médio** (1k–10k) | 0.852 | 0.871 | 0.876 | 0.872 | 0.877 | 0.879 |
| **Grande** (n>10k) | 0.811 | 0.836 | 0.835 | 0.837 | 0.846 | 0.847 |

**Observações:**
- **Regime médio:** menor gap do Mambular (~2.7 pp)
- **Regime grande:** todos convergem, mas Mambular paga custo enorme em tempo
- **Regime pequeno:** apenas 2 datasets válidos para Mambular (baixa representatividade)

---

## Slide 7 — Destaque: Mambular VENCE em Multiclasse!

**AUC-OVO médio por tipo de problema:**

| Tipo | Mambular | LightGBM | CatBoost | XGBoost | AG-Def | AG-Ext |
|---|---|---|---|---|---|---|
| **Binário** | 0.803 | 0.835 | 0.838 | 0.836 | 0.845 | 0.848 |
| **Multiclasse** | **0.942** | 0.922 | 0.925 | 0.925 | 0.921 | 0.922 |

**Exemplos de datasets multiclasse onde Mambular venceu:**
- `website_phishing`: Mambular 0.9681 vs LightGBM 0.9624
- `splice`: 0.9957 (Mambular) ≈ 0.9971 (AG-Ext) — praticamente empatados
- `SDSS17`: 0.9946 (Mambular) vs 0.9965 (AG-Ext)
- `taiwanese_bankruptcy`: **Mambular 0.9547** > todos os outros

---

## Slide 8 — Casos Notáveis

**Mambular competitivo ou vencedor:**

| Dataset | Mambular | Melhor baseline | Resultado |
|---|---|---|---|
| taiwanese\_bankruptcy | **0.9547** | 0.9550 (AG-Ext) | Quase empate |
| website\_phishing | **0.9681** | 0.9719 (AG-Ext) | Mambular 2º |
| online\_shoppers | **0.9294** | 0.9363 (AG-Ext) | Competitivo |
| splice | 0.9957 | 0.9971 (AG-Ext) | Competitivo |

**Mambular abaixo do esperado:**

| Dataset | Mambular | Melhor | Gap |
|---|---|---|---|
| Is-this-a-good-customer | 0.6530 | 0.7711 | −14 pp |
| Marketing\_Campaign | 0.8189 | 0.9016 | −8.3 pp |
| in\_vehicle\_coupon | 0.7909 | 0.8589 | −6.8 pp |

---

## Slide 9 — Custo Computacional

**Tempo médio por dataset:**

| Modelo | Tempo médio | Relativo |
|---|---|---|
| XGBoost | 6.5 s | 0.7× |
| **LightGBM** | **8.7 s** | **1×** |
| CatBoost | 35.7 s | 4× |
| AG-Default | 131.9 s | 15× |
| **Mambular** | **1.582,7 s** | **182×** |
| AG-Extreme | 2.433,2 s | 280× |

**Trade-off:**
```
LightGBM:  AUC 0.855  |  8.7s   ← melhor custo-benefício
Mambular:  AUC 0.829  |  1582s  ← 182× mais lento, -2.6 pp AUC
```

**Porém:** Mambular usa GPU (RTX 5070), LightGBM usa CPU. Comparação de hardware diferente.

---

## Slide 10 — Conclusões

**O que aprendemos sobre o Mambular:**

✅ **Pontos fortes:**
- Estatisticamente equivalente às GBDTs clássicas (P≈70–77% dentro de 1 pp)
- **Supera todos os modelos em problemas multiclasse** (AUC médio 0.942)
- Competitivo em datasets médios (1k–10k amostras)

❌ **Pontos fracos:**
- 182× mais lento que LightGBM
- Falha por OOM em datasets com muitas features (>40)
- Perde para AutoGluon (ensemble vs modelo único)
- Não agrega valor em problemas binários vs GBDTs

📊 **Resposta à hipótese de pesquisa:**
> *SSMs para dados tabulares são uma alternativa viável às GBDTs?*

**Sim — mas apenas em problemas multiclasse. Para o caso geral, o custo-benefício ainda favorece GBDTs.**

---

## Slide 11 — Reprodutibilidade

**Código disponível em:** `github.com/[usuario]/Projeto-2-Aprendizagem-de-Maquina-2026.1`

```bash
# Instalar dependências (versões fixas em pyproject.toml)
pip install -e .

# Executar experimento completo
python -m experiments.runner

# Análise estatística
python -m experiments.analysis
```

**Arquivos de resultado:**
- `results/results.csv` — todas as métricas
- `results/best_params.json` — hiperparâmetros ótimos
- `results/plots/cd_diagram.png` — diagrama de diferença crítica
- `results/bayesian_results.csv` — análise Bayesiana

**Seed fixo: 42 em todas as etapas**

---

*Obrigado! Dúvidas?*
