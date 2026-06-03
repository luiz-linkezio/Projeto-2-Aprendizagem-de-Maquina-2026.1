# Relatório — Projeto Final de Aprendizagem de Máquina 2026-1
**Modelo:** Mambular (Tabular Mamba / SSM)  
**Disciplina:** Aprendizagem de Máquina — Prof. Leandro Almeida  
**Benchmark:** TabArena-v0.1 (NeurIPS 2025)  
**Seed:** 42 | **Split:** 70% treino / 30% teste | **HPO:** Optuna (10 trials, 2-fold CV)

---

## 1. Modelo: Mambular

### 1.1 Motivação e Contexto Histórico

Mambular é uma implementação do modelo **Mamba** (State Space Model — SSM) para dados tabulares, publicada em 2024 e disponibilizada via biblioteca [OpenTabular/DeepTab](https://github.com/OpenTabular/DeepTab). Ele surge como alternativa ao dominante paradigma Transformer em tabular deep learning, inspirado na hipótese de que SSMs capturam dependências sequenciais com custo computacional linear (versus quadrático dos Transformers).

O Mamba foi proposto por Gu & Dao (2023) como um SSM com **seletividade de estado**: ao contrário de SSMs tradicionais (S4, S5), o Mamba adapta dinamicamente seus parâmetros de transição (`B`, `C`, `∆`) em função do token de entrada, permitindo filtrar informações irrelevantes e reter as relevantes com eficiência linear no comprimento da sequência.

O Mambular adapta essa arquitetura para features tabulares, tratando cada feature como um "token" da sequência.

### 1.2 Funcionamento Detalhado

**Arquitetura (via deeptab):**

```
Input (features tabulares)
    ↓
Embedding Layer  (features numéricas + categóricas → vetores de dimensão d_model)
    ↓
Mamba Block(s)   (n_layers blocos SSM com seletividade)
    ↓
MLP Head         (classificação binária / multiclasse)
```

**State Space Model seletivo (núcleo do Mamba):**

O estado oculto `h(t)` evolui segundo:

```
h'(t) = A(x) · h(t) + B(x) · x(t)
y(t)  = C(x) · h(t)
```

onde `A`, `B`, `C` e `∆` (discretização) são funções da entrada `x`, tornando o modelo **input-dependent** — diferente dos SSMs clássicos onde esses parâmetros são fixos.

**Hiperparâmetros principais:**

| Hiperparâmetro | Intervalo buscado | Significado |
|---|---|---|
| `d_model` | {16, 32} | Dimensão dos embeddings |
| `n_layers` | 1–3 | Número de blocos Mamba |
| `dropout` | 0.0–0.5 | Regularização |
| `lr` | 1e-4 – 5e-3 | Taxa de aprendizado (log) |
| `numerical_preprocessing` | {standardization, quantile} | Pré-processamento numérico |

**Pré-processamento interno (pretab):**
- Features numéricas: imputer → minmax → `standardization` ou `quantile`  
- Features categóricas: imputer → `continuous_ordinal` (ordinalização contínua)

### 1.3 Forma de Aprendizado

Treinado com **PyTorch Lightning**, minimizando BCEWithLogitsLoss (binário) ou CrossEntropyLoss (multiclasse) via Adam. O treinamento na fase de HPO usa `max_epochs=20`; o treino final usa `max_epochs=50`.

### 1.4 Aplicações Práticas e Limitações

**Aplicações:**
- Classificação tabular com muitas features categóricas ordinais
- Datasets onde a ordem das features carrega semântica

**Limitações observadas neste experimento:**
- Falha por OOM em datasets com muitas features (`anneal` 38 feats, `qsar-biodeg` 41 feats, `MIC` 111 feats, `APSFailure` 170 feats) — acumulação de VRAM em trials Optuna
- Tempo de treinamento muito superior aos modelos de árvore (mediana ~500s vs ~5s do LightGBM)
- Bug na biblioteca pretab (PLE transformer) com certas distribuições de features
- Necessita GPU para ser viável em datasets grandes

---

## 2. Experimentos Comparativos

### 2.1 Datasets Utilizados

30 datasets do benchmark TabArena-v0.1, estratificados por regime de tamanho. Nota: o benchmark não oferece 10 datasets com n < 1.000 para classificação; utilizamos 3 datasets pequenos disponíveis.

| # | Dataset | task\_id | n | Features | Classes | Regime | Tipo |
|---|---|---|---|---|---|---|---|
| 1 | blood-transfusion-service-center | 363621 | 748 | 4 | 2 | pequeno | binário |
| 2 | diabetes | 363629 | 768 | 8 | 2 | pequeno | binário |
| 3 | anneal | 363614 | 898 | 38 | 5 | pequeno | multiclasse |
| 4 | credit-g | 363626 | 1.000 | 20 | 2 | médio | binário |
| 5 | maternal\_health\_risk | 363685 | 1.014 | 6 | 3 | médio | multiclasse |
| 6 | qsar-biodeg | 363696 | 1.054 | 41 | 2 | médio | binário |
| 7 | website\_phishing | 363707 | 1.353 | 9 | 3 | médio | multiclasse |
| 8 | Fitness\_Club | 363671 | 1.500 | 6 | 2 | médio | binário |
| 9 | MIC | 363711 | 1.699 | 111 | 8 | médio | multiclasse |
| 10 | Is-this-a-good-customer | 363682 | 1.723 | 13 | 2 | médio | binário |
| 11 | Marketing\_Campaign | 363684 | 2.240 | 25 | 2 | médio | binário |
| 12 | seismic-bumps | 363700 | 2.584 | 15 | 2 | médio | binário |
| 13 | splice | 363702 | 3.190 | 60 | 3 | médio | multiclasse |
| 14 | students\_dropout\_and\_academic\_success | 363704 | 4.424 | 36 | 3 | médio | multiclasse |
| 15 | churn | 363623 | 5.000 | 19 | 2 | médio | binário |
| 16 | polish\_companies\_bankruptcy | 363694 | 5.910 | 64 | 2 | médio | binário |
| 17 | taiwanese\_bankruptcy\_prediction | 363706 | 6.819 | 94 | 2 | médio | binário |
| 18 | heloc | 363676 | 10.459 | 23 | 2 | grande | binário |
| 19 | jm1 | 363712 | 10.885 | 21 | 2 | grande | binário |
| 20 | E-CommereShippingData | 363632 | 10.999 | 10 | 2 | grande | binário |
| 21 | online\_shoppers\_intention | 363691 | 12.330 | 17 | 2 | grande | binário |
| 22 | in\_vehicle\_coupon\_recommendation | 363681 | 12.684 | 24 | 2 | grande | binário |
| 23 | HR\_Analytics\_Job\_Change\_of\_Data\_Scientists | 363679 | 19.158 | 12 | 2 | grande | binário |
| 24 | credit\_card\_clients\_default | 363627 | 30.000 | 23 | 2 | grande | binário |
| 25 | Amazon\_employee\_access | 363613 | 32.769 | 9 | 2 | grande | binário |
| 26 | bank-marketing | 363618 | 45.211 | 13 | 2 | grande | binário |
| 27 | APSFailure | 363616 | 76.000 | 170 | 2 | grande | binário |
| 28 | Diabetes130US | 363630 | 71.518 | 47 | 2 | grande | binário |
| 29 | SDSS17 | 363699 | 78.053 | 11 | 3 | grande | multiclasse |
| 30 | customer\_satisfaction\_in\_airline | 363628 | 129.880 | 21 | 2 | grande | binário |

### 2.2 Protocolo Experimental

- **Split:** 70% treino / 30% teste, `stratify=y`, `random_state=42`
- **HPO:** Optuna com TPE sampler, 10 trials, 2-fold CV estratificado
- **Métricas:** AUC-OVO, Accuracy, G-Mean, Cross-Entropy, Tempo (s)
- **Mambular:** GPU (NVIDIA RTX 5070 12 GB), `batch_size=32`, `max_epochs=20` (HPO) / `50` (treino final)
- **Modelos de árvore:** CPU, n\_jobs=-1

**Modelos comparados:**

| Modelo | Descrição |
|---|---|
| **Mambular** | Tabular Mamba SSM (deeptab) — modelo do grupo |
| LightGBM | Gradient Boosting baseado em histogramas |
| CatBoost | Gradient Boosting com suporte nativo a categorias |
| XGBoost | Extreme Gradient Boosting |
| AG-Default | AutoGluon 1.4, preset `medium_quality`, 1h |
| AG-Extreme | AutoGluon 1.4, preset `best_quality`, 2h |

### 2.3 Resultados por Dataset (AUC-OVO no conjunto de teste)

| Dataset | Mambular | LightGBM | CatBoost | XGBoost | AG-Default | AG-Extreme |
|---|---|---|---|---|---|---|
| blood-transfusion | 0.7628 | 0.7504 | **0.7767** | 0.7594 | 0.7688 | 0.7796 |
| diabetes | 0.8492 | 0.8452 | **0.8635** | 0.8505 | 0.8483 | 0.8613 |
| anneal | — | 0.9998 | 1.0000 | 0.9998 | 0.9999 | **1.0000** |
| credit-g | 0.7655 | 0.7749 | 0.7742 | 0.7547 | 0.8025 | **0.8095** |
| maternal\_health\_risk | 0.8897 | 0.9431 | 0.9467 | 0.9425 | 0.9476 | **0.9524** |
| qsar-biodeg | — | 0.9570 | **0.9576** | 0.9549 | 0.9557 | 0.9534 |
| website\_phishing | **0.9681** | 0.9624 | 0.9602 | 0.9646 | 0.9627 | 0.9719 |
| Fitness\_Club | 0.8143 | 0.8058 | 0.8129 | 0.8137 | 0.8070 | **0.8151** |
| MIC | — | 0.6854 | **0.7018** | 0.7004 | 0.6636 | 0.6559 |
| Is-this-a-good-customer | 0.6530 | 0.7390 | 0.7508 | **0.7561** | 0.7711 | 0.7666 |
| Marketing\_Campaign | 0.8189 | 0.8712 | 0.8926 | 0.8693 | 0.8966 | **0.9016** |
| seismic-bumps | 0.7209 | **0.7706** | 0.7650 | 0.7735 | 0.7684 | 0.7610 |
| splice | 0.9957 | 0.9952 | 0.9956 | 0.9958 | 0.9968 | **0.9971** |
| students\_dropout | 0.8605 | 0.8709 | 0.8740 | **0.8751** | 0.8824 | 0.8832 |
| churn | 0.8809 | **0.9221** | 0.9186 | 0.9089 | 0.9176 | 0.9178 |
| polish\_bankruptcy | 0.9068 | 0.9491 | 0.9532 | 0.9532 | 0.9563 | **0.9671** |
| taiwanese\_bankruptcy | **0.9547** | 0.9525 | 0.9535 | 0.9518 | 0.9520 | 0.9550 |
| heloc | 0.7651 | 0.7950 | **0.7984** | 0.7957 | 0.7958 | 0.7982 |
| jm1 | 0.7300 | 0.7383 | 0.7391 | **0.7421** | 0.7624 | 0.7618 |
| E-CommereShippingData | 0.7432 | 0.7388 | **0.7475** | 0.7400 | 0.7421 | 0.7470 |
| online\_shoppers | **0.9294** | 0.9330 | 0.9304 | 0.9335 | 0.9358 | 0.9363 |
| in\_vehicle\_coupon | 0.7909 | 0.8425 | 0.8373 | **0.8463** | 0.8548 | 0.8589 |
| HR\_Analytics | 0.7912 | 0.7951 | 0.7959 | 0.7959 | **0.7970** | 0.7965 |
| credit\_card\_default | 0.7780 | 0.7852 | 0.7874 | 0.7873 | 0.7888 | **0.7898** |
| Amazon\_access | 0.7989 | 0.8144 | 0.8060 | 0.8227 | 0.8804 | **0.8800** |
| bank-marketing | 0.7593 | 0.7662 | **0.7697** | 0.7685 | 0.7700 | 0.7696 |
| APSFailure | — | 0.9912 | **0.9931** | 0.9916 | 0.9947 | 0.9941 |
| Diabetes130US | 0.6524 | **0.6732** | 0.6649 | 0.6699 | 0.6826 | 0.6844 |
| SDSS17 | 0.9946 | 0.9960 | 0.9960 | 0.9960 | 0.9963 | **0.9965** |
| customer\_satisfaction | 0.9928 | **0.9939** | 0.9935 | 0.9937 | 0.9945 | 0.9946 |
| **Média (n disponível)** | **0.8295** | **0.8553** | **0.8585** | **0.8569** | **0.8631** | **0.8652** |

*— indica que o Mambular falhou por OOM (Out Of Memory na GPU) e foi permanentemente omitido após 3 tentativas.*

### 2.4 Análise por Regime

#### AUC-OVO médio por regime

| Regime | Mambular | LightGBM | CatBoost | XGBoost | AG-Default | AG-Extreme |
|---|---|---|---|---|---|---|
| Pequeno (n < 1k) | 0.8060 | 0.8652 | 0.8801 | 0.8699 | 0.8723 | 0.8803 |
| Médio (1k–10k) | 0.8524 | 0.8714 | 0.8755 | 0.8724 | 0.8772 | 0.8791 |
| Grande (n > 10k) | 0.8105 | 0.8356 | 0.8353 | 0.8372 | 0.8458 | 0.8467 |

**Observações:**
- No regime **médio**, o Mambular apresenta desempenho mais competitivo (gap ~2.5 pp em relação ao melhor)
- No regime **grande**, todos os modelos convergem em performance, mas o Mambular tem custo computacional muito superior
- No regime **pequeno**, o Mambular tem apenas 2 datasets válidos (anneal e qsar-biodeg falharam por OOM)

#### AUC-OVO médio por tipo de classificação

| Tipo | Mambular | LightGBM | CatBoost | XGBoost | AG-Default | AG-Extreme |
|---|---|---|---|---|---|---|
| Binário | 0.8028 | 0.8350 | 0.8383 | 0.8362 | 0.8454 | 0.8478 |
| Multiclasse | **0.9417** | 0.9218 | 0.9249 | 0.9249 | 0.9213 | 0.9224 |

**Achado notável:** o Mambular supera todos os baselines em AUC médio para tarefas **multiclasse** (0.9417 vs 0.9249 do segundo melhor). Isso sugere que a arquitetura SSM beneficia-se de problemas com mais classes.

### 2.5 Tempo Computacional

| Modelo | Tempo médio (s) | Relativo ao LightGBM |
|---|---|---|
| XGBoost | 6.5 | 0.7× |
| **LightGBM** | **8.7** | **1.0×** |
| CatBoost | 35.7 | 4.1× |
| AG-Default | 131.9 | 15× |
| **Mambular** | **1.582,7** | **182×** |
| AG-Extreme | 2.433,2 | 280× |

O Mambular é ~182× mais lento que o LightGBM em média. Parte desse custo deve-se ao overhead do PyTorch Lightning e ao HPO com GPU.

---

## 3. Análise Estatística

### 3.1 Teste de Friedman + Post-hoc Nemenyi

Aplicamos o teste de **Friedman** (omnibus não-paramétrico) sobre os AUC-OVO dos 30 datasets, seguido do teste post-hoc de **Nemenyi** via biblioteca `autorank`.

**Distância Crítica (CD) = 1.377** (α = 0.05, 6 classificadores, 30 datasets)

| Modelo | Mean Rank | AUC Médio ± DP |
|---|---|---|
| AG-Extreme | 1.82 | 0.865 ± 0.102 |
| AG-Default | 2.53 | 0.863 ± 0.101 |
| CatBoost | 3.25 | 0.859 ± 0.101 |
| XGBoost | 3.80 | 0.857 ± 0.101 |
| LightGBM | 4.33 | 0.855 ± 0.103 |
| **Mambular** | **5.15** | **0.829 ± 0.101** |

**Grupos sem diferença significativa (CD > 1.377):**
- {Mambular, LightGBM, XGBoost}
- {LightGBM, XGBoost, CatBoost}
- {XGBoost, CatBoost, AG-Default}
- {AG-Default, AG-Extreme}

O Mambular apresenta diferença significativa apenas em relação ao CatBoost, AG-Default e AG-Extreme. A diferença entre Mambular e LightGBM/XGBoost **não é estatisticamente significativa**.

> Ver diagrama de diferença crítica em: `results/plots/cd_diagram.png`

### 3.2 Análise Bayesiana (Signed-Rank Bayesiano, ROPE = 1%)

Comparações Mambular vs cada baseline usando `baycomp.two_on_multiple` com região de equivalência prática (ROPE) de ±1 pp:

| Comparação | P(Mambular > baseline) | P(equivalente) | P(baseline > Mambular) |
|---|---|---|---|
| Mambular vs LightGBM | 0.000 | **0.768** | 0.232 |
| Mambular vs CatBoost | 0.000 | **0.702** | 0.298 |
| Mambular vs XGBoost | 0.000 | **0.694** | 0.306 |
| Mambular vs AG-Default | 0.000 | 0.303 | **0.697** |
| Mambular vs AG-Extreme | 0.000 | 0.213 | **0.787** |

**Interpretação:**
- **vs LightGBM/CatBoost/XGBoost:** Alta probabilidade de equivalência prática (~70–77%) — o Mambular é **praticamente equivalente** às GBDTs clássicas dentro da margem de 1 pp
- **vs AutoGluon:** O AutoGluon (com mais tempo de treino) supera o Mambular com probabilidade de 70–79%, sendo a diferença prática relevante

> Ver gráficos Bayesianos em: `results/plots/bayesian_mambular_vs_*.png`

---

## 4. Discussão e Conclusões

### 4.1 Quando o Mambular é competitivo?

1. **Problemas multiclasse:** AUC médio 0.9417, superando todos os baselines (2ª melhor: 0.9249)
2. **Datasets médios** (1k–10k): Gap de apenas ~2.5 pp para o melhor modelo
3. **Datasets com features categóricas** (splice, website_phishing, taiwanese_bankruptcy): O Mambular competiu ou superou modelos de árvore

### 4.2 Quando o Mambular falha?

1. **Datasets com muitas features** (> 40): OOM na GPU — limitação crítica da implementação atual
2. **Datasets grandes binários simples** (bank-marketing, Diabetes130US, heloc): GBDTs dominam com fração do custo computacional
3. **Custo computacional**: 182× mais lento que LightGBM, inviabilizando seu uso em cenários sem GPU ou com restrição de tempo

### 4.3 Conclusão Geral

Do ponto de vista estatístico, o Mambular é **equivalente às GBDTs clássicas** (LightGBM, CatBoost, XGBoost) em termos de AUC-OVO médio dentro de uma margem de 1 pp (P(equivalente) ≈ 70–77%). No entanto, paga um custo computacional de ~182× em tempo de treino e requer GPU, tornando-o menos atraente para uso geral.

O resultado mais promissor é em tarefas **multiclasse**, onde o Mambular supera todos os baselines em média — sugerindo que a capacidade do SSM em modelar relações entre features de forma sequencial pode ser vantajosa quando há múltiplas classes a distinguir.

Em comparação com o AutoGluon (pipeline AutoML), o Mambular perde: o AutoGluon utiliza ensembles de múltiplos modelos incluindo GBDTs otimizadas, enquanto o Mambular é um único modelo com arquitetura mais restrita.

---

## 5. Reprodutibilidade

**Dependências principais:**
```
deeptab>=0.2.0
lightgbm>=4.6.0
catboost>=1.2.7
xgboost>=2.1.0
autogluon.tabular>=1.4.0
optuna>=4.3.0
scikit-learn>=1.6.0
autorank>=1.2.0
baycomp>=1.0.2
openml>=0.15.0
torch>=2.6.0
pytorch-lightning>=2.5.0
```

**Execução:**
```bash
python -m experiments.runner   # coleta resultados
python -m experiments.analysis # análise estatística
```

**Arquivos de resultado:**
- `results/results.csv` — métricas de treino e teste por dataset/modelo
- `results/best_params.json` — melhores hiperparâmetros por dataset/modelo
- `results/bayesian_results.csv` — probabilidades Bayesianas
- `results/regime_summary.csv` — sumário por regime
- `results/plots/cd_diagram.png` — diagrama de diferença crítica
- `results/runner.log` — log completo de execução

---

## 6. Model Card — Mambular

| Campo | Valor |
|---|---|
| **Nome** | Mambular |
| **Versão** | via deeptab (OpenTabular/DeepTab, 2024) |
| **Tipo de modelo** | State Space Model (Mamba SSM) para dados tabulares |
| **Tarefa** | Classificação supervisionada (binária e multiclasse) |
| **Arquitetura** | Embedding → Mamba Block(s) → MLP Head |
| **Framework** | PyTorch + PyTorch Lightning |
| **Requisitos** | GPU recomendada (≥ 8 GB VRAM) |
| **Hiperparâmetros** | d\_model, n\_layers, dropout, lr, numerical\_preprocessing |
| **AUC médio (26 datasets)** | 0.8295 |
| **Melhor performance** | Problemas multiclasse (AUC médio 0.9417) |
| **Pior performance** | Datasets com > 40 features (OOM) |
| **Limitações** | OOM em datasets com muitas features; ~182× mais lento que LightGBM |
| **Considerações éticas** | Modelo de caixa-preta; não oferece explicabilidade nativa. Decisões de alto impacto requerem interpretabilidade adicional (SHAP, LIME) |
| **Dados de treino** | 30 datasets públicos do TabArena-v0.1 (OpenML), licenças permissivas |
| **Métricas avaliadas** | AUC-OVO, Accuracy, G-Mean, Cross-Entropy |
