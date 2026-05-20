# Multi-Agent System for Social Data Analysis

A heterogeneous stacking ensemble for financial sentiment classification.
Combines four fundamentally different model paradigms — TF-IDF + logistic
regression, fine-tuned FinBERT, GPT-4o-mini (zero-shot), and the
Loughran-McDonald financial lexicon — and reads their soft probabilities into
a logistic-regression meta-learner trained on the validation split. An
optional fifth agent (fine-tuned FinTwitBERT) lifts the stack a further
0.5 – 4.0 percentage points of accuracy on three of the four labelled corpora.

This is the code companion to the master's thesis
*"Multi-Agent System for Social Data Analysis"* (HSE, Faculty of Computer
Science, MSc Data Science programme, 2026). Reproducing every figure and
table in the thesis is the explicit goal of this repository.

## Headline result

|              | Twitter | PhraseBank | SemEval-2017 | FiQA-2018 |
|--------------|--------:|-----------:|-------------:|----------:|
| FinTwitBERT-alone   | 0.9028 | 0.8556 | 0.8160 | 0.8409 |
| **5-agent stack**   | **0.9079** | **0.8886** | **0.8208** | **0.8523** |
| ΔPP (5-stack − FinTwit) | +0.51 | +3.30 | +0.48 | +1.14 |
| McNemar p           | 0.42 | 0.0014 | 0.84 | 0.77 |

Per-corpus calibration, selective prediction, disagreement-subset analysis
and a downstream backtest on 3,317 FNSPID news headlines (4 horizons × 2
benchmarks) are produced by the scripts in `scripts/`. See
[`RUN_AND_DEPLOY.md`](RUN_AND_DEPLOY.md) for the exact order of operations.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │                 Input text                │
                    └────────────────────┬─────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
       ┌──────▼──────┐          ┌────────▼────────┐         ┌──────▼──────┐
       │ TF-IDF + LR │          │  FinBERT (FT)   │         │ GPT-4o-mini │
       └──────┬──────┘          └────────┬────────┘         └──────┬──────┘
              │                          │                          │
              │              ┌───────────▼───────────┐              │
              │              │ Loughran–McDonald lex │              │
              │              └───────────┬───────────┘              │
              │                          │                          │
              │              ┌───────────▼───────────┐ (optional)   │
              │              │ FinTwitBERT (FT)      │              │
              │              └───────────┬───────────┘              │
              └──────────────────────────┼──────────────────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │  Logistic-regression        │
                          │  meta-learner               │
                          │  (12 or 15 input features,  │
                          │   trained on val split)     │
                          └──────────────┬──────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │  3-class output             │
                          │  {negative, neutral,        │
                          │   positive} + calibrated    │
                          │  probabilities              │
                          └─────────────────────────────┘
```

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/olivan139/multiagent-system-finance.git
cd multiagent-system-finance

# 2. Create a virtual env (Python 3.11+)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install the package + every optional dependency group
pip install -e ".[all]"

# 4. Configure the LLM key (only needed for the GPT-4o-mini agent and
#    the multi-agent pipeline)
cp .env.example .env
# Edit .env and paste your OPENAI_API_KEY

# 5. Smoke-check the classical baselines (no API key, no GPU, < 30 s)
python scripts/run_phase1_classical.py
```

For the full reproduction recipe (datasets, transformer fine-tuning, agent
caches, 5-agent stacking, calibration, FNSPID downstream backtest) read
[`RUN_AND_DEPLOY.md`](RUN_AND_DEPLOY.md) end-to-end.

## Datasets

The ensemble is evaluated on four publicly available labelled corpora:

| Corpus | Source | n_train | n_val | n_test |
|--------|--------|--------:|------:|-------:|
| Twitter Financial News Sentiment | `zeroshot/twitter-financial-news-sentiment` (HF) | 9,543 | 1,194 | 1,194 |
| Financial PhraseBank (75 % agreement) | `financial_phrasebank` (HF) | 3,393 | 727 | 727 |
| SemEval-2017 Task 5 | sponsor distribution | ≈ 1,700 | 423 | 424 |
| FiQA-2018 Task 1   | sponsor distribution | ≈ 700 | 176 | 176 |

A fifth unlabelled corpus, **FNSPID** (Financial News and Stock Price
Integration Dataset), is used downstream for the signal-strength test
described in `RUN_AND_DEPLOY.md` § 6.

## Repository layout

```
multiagent-system-finance/
├── mas/                            # Core library
│   ├── config.py                   # Labels, paths, dataclasses
│   ├── agents/
│   │   ├── base.py                 # Common agent interface
│   │   ├── single.py               # Single-call GPT-4o-mini agent
│   │   ├── multi.py                # Analyst → Fact-checker → Aggregator
│   │   ├── debate.py               # Generator → Discriminator → Arbiter
│   │   └── ensemble.py             # Heterogeneous stacking ensemble
│   ├── baselines/
│   │   ├── tfidf_logreg.py
│   │   ├── transformer.py          # FinBERT (zero-shot + fine-tuning)
│   │   └── lexicon.py              # Loughran-McDonald (pysentiment2)
│   ├── data/
│   │   ├── loader.py               # HF, CSV, PhraseBank, SemEval, FiQA
│   │   ├── preprocessing.py
│   │   ├── fnspid.py               # FNSPID corpus loader
│   │   └── sp500_universe.py
│   └── evaluation/
│       ├── metrics.py              # Accuracy / F1 / Cohen's κ / CM plots
│       └── statistical.py          # McNemar, bootstrap, Friedman tests
├── scripts/                        # CLI entry points (run in any order
│                                   #   subject to the dependency graph
│                                   #   in RUN_AND_DEPLOY.md)
├── dashboard.py                    # Streamlit results dashboard
├── pyproject.toml
├── requirements.txt
├── .env.example
├── LICENSE
├── README.md
└── RUN_AND_DEPLOY.md
```
