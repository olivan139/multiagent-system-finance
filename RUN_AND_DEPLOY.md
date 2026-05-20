# Run & Deploy

End-to-end instructions for reproducing every model, table and figure from
the thesis, and for running the ensemble in inference / batch mode.

The pipeline is intentionally split into small, idempotent steps so any
single step can be re-run without invalidating the others. Each step writes
its outputs into `results/` and the next step reads from `results/`.

> All commands assume you are at the repository root with the virtual env
> active and the `mas-sentiment` package installed in editable mode
> (`pip install -e ".[all]"`).

---

## 0. Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (only for steps 2, 3, 4)
```

Minimum hardware requirements:

| Step | RAM | GPU | Disk |
|------|----:|:---:|----:|
| 1, 5, 7, 8, 9 (classical / stats / analysis) | 4 GB | not needed | 1 GB |
| 2 (FinBERT fine-tuning), 3 (FinTwitBERT fine-tuning) | 12 GB | recommended (CUDA or MPS); CPU works in ≈ 1 h per dataset | 4 GB |
| 6 (FNSPID downstream) | 8 GB | not needed | 15 GB (raw FNSPID corpus + 21 yfinance parquet) |
| 4 (LLM inference) | 4 GB | not needed | OpenAI API budget (≈ \$0.05 per 1 000 samples) |

---

## 1. Datasets

Three of the four labelled corpora download automatically through Hugging
Face the first time they are loaded:

* `zeroshot/twitter-financial-news-sentiment` (Twitter)
* `financial_phrasebank` config `sentences_75agree` (PhraseBank)

The other two require a one-time manual placement under `data/`:

```
data/
├── semeval2017_task5/
│   ├── train.json          # SemEval-2017 Task 5 train
│   └── test.json           # SemEval-2017 Task 5 test
└── fiqa2018/
    ├── train.json          # FiQA-2018 Task 1 train
    └── test.json           # FiQA-2018 Task 1 test
```

Both are available from their respective sponsor sites (see
`mas/data/loader.py` for the exact JSON schema we expect).

For the downstream FNSPID test (step 6 only) you need
`data/fnspid/all_external_first_50000000.csv` from
[`https://huggingface.co/datasets/Zihan1004/FNSPID`](https://huggingface.co/datasets/Zihan1004/FNSPID).
This is ≈ 5 GB and is required only for step 6.

---

## 2. Phase-1 classical baselines (no API key, no GPU)

```bash
python scripts/run_phase1_classical.py             # Twitter only
python scripts/run_phase1_classical.py --dataset phrasebank
```

Produces:

* `results/{dataset}/phase1_results.json` – TF-IDF + LogReg metrics
* `results/{dataset}/_cache_agents.npz` – partial agent cache (TF-IDF slots)

---

## 3. Phase-2 transformer + LLM baselines

```bash
# FinBERT fine-tuning (writes results/finbert_finetuned_<dataset>/)
python scripts/run_phase2_transformer_llm.py --dataset twitter
python scripts/run_phase2_transformer_llm.py --dataset phrasebank

# FinBERT zero-shot only (no fine-tune, no LLM)
python scripts/run_phase2_transformer_llm.py --skip-finetune --skip-llm

# Smoke test on 50 samples
python scripts/run_phase2_transformer_llm.py --max-samples 50
```

The fine-tuning script defaults to MPS on Apple-Silicon, CUDA on Linux. It
writes a Hugging Face checkpoint into
`results/finbert_finetuned_<dataset>/`, plus a serialised
`_cache_finbert.npz` with the per-sample val + test probabilities.

---

## 4. Build full agent caches for the 4 labelled datasets

These scripts run all four agents (TF-IDF, FinBERT-FT, GPT-4o-mini,
Loughran-McDonald) on validation + test and persist the per-sample
probabilities. The downstream stacking, calibration, McNemar and bootstrap
scripts all read from these caches; you only need to run them once per
dataset.

```bash
python scripts/build_phrasebank_cache.py
python scripts/build_semeval_cache.py
python scripts/build_fiqa_cache.py
# Twitter caches are produced by step 2.
```

To refresh just the lexicon column after a Loughran-McDonald change:

```bash
python scripts/regen_lexicon_cache.py --dataset phrasebank
python scripts/refresh_twitter_lexicon.py
```

For the optional fifth agent (FinTwitBERT, used in §5.10 of the thesis):

```bash
python scripts/fit_fintwitbert_agent.py --dataset twitter
python scripts/fit_fintwitbert_agent.py --dataset phrasebank
python scripts/fit_fintwitbert_agent.py --dataset semeval2017
python scripts/fit_fintwitbert_agent.py --dataset fiqa2018
```

---

## 5. Refit the stacking meta-learner and regenerate all metrics

This step recomputes every row of `all_results.json` and every entry of
`predictions.json` and `statistical_report.json` from the caches built in
step 4. Anything downstream (calibration analysis, McNemar table, bootstrap
table, agent agreement heat-map, error analysis) reads from these.

```bash
# 4-agent stack + per-agent + 4 ablations + majority + weighted-average
python scripts/regen_predictions_and_stats.py            # twitter + phrasebank

# 5-agent stack (adds FinTwitBERT as 5th input feature)
python scripts/refit_five_agent_stack.py                 # all four datasets

# Alternative: meta-learner = XGBoost (vs the production LogReg)
python scripts/xgboost_meta_tuned.py                     # 80-trial CV search
python scripts/xgboost_meta_comparison.py                # head-to-head
```

After this step the contents of `results/` are internally consistent: every
metric in every JSON file is computed from the same per-sample cache.

---

## 6. Calibration, selective prediction, error analysis

```bash
# Per-agent temperature scaling and ECE / NLL on Twitter
python scripts/calibration_analysis.py

# 5-stack vs FinTwit-alone: Brier / NLL / ECE / selective prediction /
# disagreement subset across all four datasets
python scripts/calibration_selective_disagreement.py

# Per-bucket error analysis (ticker / digits / length / sentiment-word density)
python scripts/error_analysis.py

# Cross-dataset comparison summary
python scripts/cross_dataset_analysis.py
```

---

## 7. Downstream FNSPID signal test

The thesis' fifth corpus is FNSPID, the Financial News and Stock Price
Integration Dataset. The downstream test joins 3,317 sampled headlines
(or 28,288 in the broader-universe variant) to 21 yfinance abnormal-return
series at four horizons (1d, 5d, 20d, 30d) × two benchmarks (SPY, sector).

```bash
# 1. Download yfinance prices (≈ 30 s, requires internet)
python scripts/fetch_fnspid_prices.py

# 2. Score every FNSPID headline with the 4-agent ensemble and the
#    standalone agents (this is the long step, ≈ 90 min for 3,317 rows
#    with LLM in the loop; the no-LLM variant runs in 5 min)
python scripts/run_fintwit_on_fnspid.py
python scripts/patch_fnspid_predictions.py   # back-fills any missed rows

# 3. Compute IC / pos-neg spread / Welch p / bootstrap CIs across
#    every (system × horizon × benchmark) combination
python scripts/downstream_metrics.py

# 4. Optional: also run the classical, frequentist long-short signal test
python scripts/downstream_signal_test.py

# 5. Generate the figures used in Section 5.X of the thesis
python scripts/make_downstream_figures.py --fig-dir figures/
```

---

## 8. Reproduce the multi-agent LLM pipelines (for comparison)

These are the LLM-only baselines the thesis compares against the
heterogeneous ensemble.

```bash
# Analyst → Fact-Checker → Aggregator (≈ \$0.0005/sample, ≈ 8.6 s/sample)
python scripts/run_phase3_multi_agent.py
python scripts/run_phase3_multi_agent.py --max-samples 50   # smoke test

# Generator → Discriminator → Arbiter with confidence-routed debate
python scripts/run_phase4_debate_ablation.py
python scripts/run_phase4_debate_ablation.py --skip-ablation
```

---

## 9. Re-run absolutely everything in one go

For full reproduction from a clean clone (≥ 12 GB disk and a few hours
of compute):

```bash
python scripts/run_all_phases.py                   # everything
python scripts/run_all_phases.py --max-samples 20  # quick validation
```

---

## 10. Interactive results dashboard

```bash
pip install streamlit
streamlit run dashboard.py
```

The dashboard reads `results/*/all_results.json` and presents the
accuracy / F1 / Kappa comparison table plus the confusion matrices.

---

## 11. Inference on your own data

```python
from mas.baselines.tfidf_logreg import TfidfLogRegBaseline
from mas.baselines.transformer import TransformerBaseline
from mas.baselines.lexicon import LoughranMcDonaldAgent
from mas.agents.single import SingleLLMAgent
from mas.agents.ensemble import HeterogeneousEnsemble

# 1. Load / train the four base agents on your own train+val split
tfidf = TfidfLogRegBaseline(); tfidf.train(train_texts, train_labels)
finbert = TransformerBaseline("ProsusAI/finbert").fit(train_texts, train_labels)
llm = SingleLLMAgent()                       # uses OPENAI_API_KEY
lex = LoughranMcDonaldAgent()                # no training needed

# 2. Wire them into the ensemble
ens = HeterogeneousEnsemble(tfidf, finbert, llm, lexicon_agent=lex,
                            strategy="stacking")

# 3. Fit the meta-learner on the validation split
ens.fit_meta_learner(val_texts, val_labels)

# 4. Predict on new data
predictions = ens.predict_labels(new_texts)
```

---

## Troubleshooting

* **`OSError: model not found` for `ProsusAI/finbert`** — make sure
  `pip install -e ".[transformers]"` succeeded and that you have internet
  access for the first model download.
* **OpenAI 429 / 401** — set `OPENAI_API_KEY` in `.env` and check the API
  account has enough credit. The single-LLM and multi-agent pipelines are
  the only steps that call the API.
* **`results/{dataset}/_cache_*.npz` missing** — re-run the corresponding
  `build_*_cache.py` script (step 4) for that dataset.
* **`FileNotFoundError: data/fnspid/...`** — the FNSPID corpus is required
  only for step 6. Download it from Hugging Face
  (`Zihan1004/FNSPID`, the `all_external_first_50000000.csv` shard) and
  place it under `data/fnspid/`.
