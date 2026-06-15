# Slides — Etapa 1: Apresentação Teórica do Mambular
**Projeto Final AM 2026-1 | Prof. Leandro Almeida | 10 minutos**

---

## Slide 1 — Título

# Mambular
### Tabular Mamba: State Space Models para Dados Tabulares

**Fonte:** OpenTabular/DeepTab (2024)  
GitHub: https://github.com/OpenTabular/DeepTab

---

## Slide 2 — Motivação: O Problema dos Dados Tabulares

**Por que dados tabulares são difíceis para Deep Learning?**

- Features heterogêneas (numéricas + categóricas)
- Sem estrutura espacial (imagens) ou sequencial (texto)
- Gradient Boosted Trees (LightGBM, XGBoost) dominam há anos

**Progressão histórica:**
```
TabNet (2019) → FT-Transformer (2021) → TabPFN (2022)
       → Mamba (2023) → Mambular (2024)
```

**Pergunta central:**
> Modelos de linguagem sequenciais conseguem aprender padrões tabulares?

---

## Slide 3 — O que é um State Space Model (SSM)?

**Inspiração:** Sistemas de controle linear em tempo contínuo

```
h'(t) = A · h(t) + B · x(t)      ← evolução do estado oculto
 y(t) = C · h(t)                  ← saída
```

**Discretização (para processamento digital):**
```
h[t] = Ā · h[t-1] + B̄ · x[t]
y[t] = C  · h[t]
```

**Análogo a:**
- RNN com memória estruturada
- Convolução eficiente no domínio do tempo

**Vantagem sobre Transformers:** Complexidade **O(n)** vs **O(n²)**

---

## Slide 4 — Mamba: SSM com Seletividade

**Problema dos SSMs clássicos (S4, S5):**
> Parâmetros A, B, C são **fixos** — o modelo não consegue ignorar informações irrelevantes

**Solução do Mamba (Gu & Dao, 2023):**

```
B(x) = Linear(x)     ← parâmetros dependem da entrada!
C(x) = Linear(x)
Δ(x) = softplus(Linear(x))   ← discretização adaptativa
```

**Resultado:** O modelo aprende **o que memorizar e o que esquecer** em função do input atual

> Analogia: como atenção, mas em O(n)

---

## Slide 5 — Arquitetura do Mambular

**Do Mamba para dados tabulares:**

```
┌─────────────────────────────────────────────┐
│  Input: features tabulares                  │
│         (numéricas + categóricas)           │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  Embedding Layer                            │
│  • Numéricas: imputer → minmax → preproc    │
│  • Categóricas: imputer → ordinal contínua  │
│  • Saída: sequência de vetores (d_model)    │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  Mamba Block × n_layers                     │
│  • Expansão → SSM seletivo → Compressão     │
│  • Residual connection + LayerNorm          │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  MLP Head → Softmax / Sigmoid               │
└─────────────────────────────────────────────┘
```

**Cada feature é um "token" da sequência!**

---

## Slide 6 — Hiperparâmetros Principais

| Hiperparâmetro | Intervalo | Significado |
|---|---|---|
| `d_model` | {16, 32} | Dimensão dos embeddings de feature |
| `n_layers` | 1–3 | Profundidade (blocos Mamba empilhados) |
| `dropout` | 0.0–0.5 | Regularização por dropout |
| `lr` | 1e-4 – 5e-3 | Taxa de aprendizado (Adam) |
| `numerical_preprocessing` | {standardization, quantile} | Normalização numérica |

**Busca de hiperparâmetros:**
- Optuna com TPE Sampler
- 10 trials, 2-fold CV estratificado
- Otimiza: AUC-OVO no conjunto de validação

---

## Slide 7 — Como o Mambular Aprende?

**Fase de treino (PyTorch Lightning):**

```
Para cada mini-batch (batch_size=32):
  1. Embedding de todas as features
  2. Passa pelos blocos Mamba (forward pass)
  3. Calcula loss:
     • Binário:      BCEWithLogitsLoss
     • Multiclasse:  CrossEntropyLoss
  4. Backprop (Adam, gradient clipping)
  5. Atualiza parâmetros A, B, C, Δ e embeddings
```

**O SSM aprende a "ordem de importância" das features:**
- Features mais informativas → estado mais retido
- Features irrelevantes → descartadas pelo gate seletivo

**HPO:** 20 épocas por trial  
**Treino final:** 50 épocas com melhores hiperparâmetros

---

## Slide 8 — Representação dos Padrões nos Dados

**Como o Mambular representa o conhecimento:**

```
Feature₁ → embedding₁ → h₁ ──┐
Feature₂ → embedding₂ → h₂ ──┤→ estado acumulado → predição
Feature₃ → embedding₃ → h₃ ──┤
...                            │
Featureₙ → embeddingₙ → hₙ ──┘
```

- **Embeddings numéricos:** capturam magnitude e distribuição
- **Embeddings categóricos:** ordinalização contínua (preserva semântica ordinal)
- **Estado oculto:** memória comprimida de todas as features anteriores
- **Gate seletivo (Δ):** decide quanta informação "passa" para o próximo estado

**Diferença do Transformer:** sem atenção pairwise — cada feature "vê" apenas o histórico de features anteriores na sequência

---

## Slide 9 — Aplicações Práticas

**Domínios onde o Mambular é aplicável:**

| Domínio | Exemplo |
|---|---|
| Finanças | Detecção de inadimplência, fraude |
| Saúde | Risco clínico, diagnóstico |
| E-commerce | Churn, satisfação de clientes |
| Astronomia | Classificação de objetos celestes (SDSS17) |
| Manufatura | Falha de equipamentos (APSFailure) |

**Vantagem potencial:** datasets onde features têm ordenação semântica natural (ex: questionários, escalas ordinais)

---

## Slide 10 — Limitações do Mambular

**Limitações identificadas na literatura e neste experimento:**

| Limitação | Descrição |
|---|---|
| **Custo computacional** | ~182× mais lento que LightGBM em média |
| **Requer GPU** | Inviável em CPU para datasets grandes |
| **OOM em alta dimensionalidade** | Falha com > 40 features (acumulação de VRAM) |
| **Caixa-preta** | Sem explicabilidade nativa (necessita SHAP/LIME) |
| **Bug de biblioteca** | PLE transformer falha com certas distribuições |
| **Sem vantagem em binário** | GBDTs dominam com fração do custo |

**Conclusão da Etapa 1:**
> O Mambular é uma arquitetura promissora — especialmente para problemas **multiclasse** — mas ainda não supera GBDTs clássicas no geral, com custo computacional muito superior.

---

*Próxima etapa: Resultados experimentais nos 30 datasets do TabArena-v0.1*
