#!/usr/bin/env python3
"""Section 5.X: Downstream Signal Test on FNSPID.

End-to-end orchestrator that:
  1. Loads a sliced FNSPID dataset (date+ticker filtered).
  2. Re-uses the Twitter-trained agents (TF-IDF + LogReg, fine-tuned
     FinBERT, GPT-4o-mini, Loughran-McDonald lexicon).
  3. Runs all 4 agents on FNSPID headlines (LLM with thread-pool concurrency).
  4. Re-fits the stacking meta-learner from the cached Twitter validation
     predictions (free, deterministic) and applies it on FNSPID.
  5. Persists predictions + per-agent probability vectors to
     ``results/fnspid/predictions.json``.

Output schema (predictions.json)
--------------------------------
{
    "meta": {...slice provenance + LLM cost...},
    "rows": [
        {
            "text", "ticker", "date",
            "tfidf_proba", "finbert_proba", "llm_proba", "lex_proba",
            "ensemble_proba",
            "tfidf_pred", "finbert_pred", "llm_pred", "lex_pred", "ensemble_pred",
            "llm_confidence"
        },
        ...
    ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.agents.ensemble import _llm_to_pseudo_proba
from mas.agents.single import SingleLLMAgent
from mas.baselines.lexicon import LoughranMcDonaldAgent
from mas.baselines.tfidf_logreg import TfidfLogRegBaseline
from mas.baselines.transformer import TransformerBaseline
from mas.config import LABEL2ID, LABELS, RESULTS_DIR, DataConfig
from mas.data import (
    DEFAULT_TICKER_WHITELIST,
    load_financial_phrasebank,
    load_fnspid_full,
    load_fnspid_slice,
)
from mas.data.fnspid import LEGACY_AB_WHITELIST
from mas.data.preprocessing import preprocess_batch

CACHE_TWITTER_AGENTS = RESULTS_DIR / "twitter" / "_cache_agents.npz"
CACHE_TWITTER_FINBERT = RESULTS_DIR / "twitter" / "_cache_finbert.npz"
CACHE_PB_AGENTS = RESULTS_DIR / "phrasebank" / "_cache_agents.npz"
CACHE_PB_FINBERT = RESULTS_DIR / "phrasebank" / "_cache_finbert.npz"


FT_FINBERT_TWITTER_DIR = RESULTS_DIR / "finbert_finetuned"
FT_FINBERT_PB_DIR = RESULTS_DIR / "finbert_finetuned_phrasebank"

OUT_DIR = RESULTS_DIR / "fnspid"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_llm_concurrent(
    agent: SingleLLMAgent,
    texts: list[str],
    n_workers: int = 8,
    cache_path: Path | None = None,
) -> tuple[list[str], np.ndarray]:
    """Run the LLM agent concurrently over ``texts``; resumable via cache.

    Cache layout: a single .json file mapping ``str(index) -> {label, conf}``.
    Resuming reuses any rows already done and only sends new requests.
    """
    n = len(texts)
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"  [LLM] resuming with {len(cache)} cached rows out of {n}")

    todo = [i for i in range(n) if str(i) not in cache]
    if todo:
        print(f"  [LLM] querying GPT-4o-mini on {len(todo)} rows ({n_workers} workers)")
        t0 = time.time()
        done = 0

        def _one(i: int) -> tuple[int, str, float]:
            r = agent.analyze(texts[i])
            return i, r.sentiment, float(r.confidence)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for fut in ex.map(_one, todo):
                i, lab, conf = fut
                cache[str(i)] = {"label": lab, "confidence": conf}
                done += 1
                if done % 100 == 0:
                    rate = done / (time.time() - t0 + 1e-9)
                    eta = (len(todo) - done) / max(rate, 1e-9)
                    print(
                        f"    [LLM] {done}/{len(todo)} done "
                        f"({rate:.1f} req/s, ETA {eta/60:.1f} min)"
                    )
                if cache_path and done % 250 == 0:
                    cache_path.write_text(json.dumps(cache))
        if cache_path:
            cache_path.write_text(json.dumps(cache))
        print(
            f"  [LLM] finished in {(time.time() - t0)/60:.1f} min "
            f"(cost ${agent.total_cost_usd:.4f}, {agent.total_tokens} tokens)"
        )

    labels = [cache[str(i)]["label"] for i in range(n)]
    confs = np.array([cache[str(i)]["confidence"] for i in range(n)], dtype=np.float64)
    return labels, confs


def _tfidf_proba_aligned(model: TfidfLogRegBaseline, texts: list[str]) -> np.ndarray:
    """Run TF-IDF predict_proba and re-index columns to canonical ``LABELS``."""
    raw = model.predict_proba(texts)
    classes = list(model.pipeline.classes_)
    out = np.zeros((len(texts), len(LABELS)), dtype=np.float64)
    for j, lab in enumerate(LABELS):
        if lab in classes:
            out[:, j] = raw[:, classes.index(lab)]
    return out


def _fit_meta_from_cache(
    meta_source: str = "twitter", random_seed: int = 42, C: float = 1.0, drop_llm: bool = False
):
    """Fit the stacking meta-learner from a cached agent-prob bundle.

    ``meta_source`` controls which cache to read:
      - ``twitter``:  Twitter Financial News Sentiment validation split
                      (informal / social-media domain).
      - ``phrasebank``: Financial PhraseBank validation split
                       (formal news-headline domain — matches FNSPID).

    When ``drop_llm`` is True the LLM column is excluded from the feature
    stack, producing a 9-feature 3-agent meta-learner. The test-side
    feature stack must match this geometry.
    """
    from sklearn.linear_model import LogisticRegression

    if meta_source == "twitter":
        ag_path, fb_path, label_key = (CACHE_TWITTER_AGENTS, CACHE_TWITTER_FINBERT, "twitter_val")
    elif meta_source == "phrasebank":
        ag_path, fb_path, label_key = (CACHE_PB_AGENTS, CACHE_PB_FINBERT, "phrasebank_val")
    else:
        raise ValueError(f"unknown meta_source={meta_source!r}")

    if not ag_path.exists() or not fb_path.exists():
        raise FileNotFoundError(
            f"{meta_source} cache not found at {ag_path} / {fb_path}.\n"
            f"  Build it with scripts/build_phrasebank_cache.py "
            f"(phrasebank) or scripts/_fix_part_b_other_agents.py (twitter)."
        )

    ag = np.load(ag_path, allow_pickle=True)
    fb = np.load(fb_path, allow_pickle=True)

    feature_blocks = [ag["tfidf_val_proba"], fb["val_proba"]]
    agent_order = ["tfidf", "finbert"]
    if not drop_llm:
        feature_blocks.append(ag["llm_val_proba"])
        agent_order.append("llm")
    feature_blocks.append(ag["lex_val_proba"])
    agent_order.append("lex")

    val_X = np.hstack(feature_blocks).astype(np.float64)
    val_y = np.array([LABEL2ID[lab] for lab in ag["val_labels"]])

    meta = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=random_seed)
    meta.fit(val_X, val_y)
    train_acc = float(meta.score(val_X, val_y))
    return meta, {
        "meta_source": meta_source,
        "agent_order": agent_order,
        "drop_llm": bool(drop_llm),
        f"train_accuracy_on_{label_key}": train_acc,
        "n_features": val_X.shape[1],
        "n_train_samples": val_X.shape[0],
    }


_fit_meta_from_twitter_cache = lambda **kw: _fit_meta_from_cache(meta_source="twitter", **kw)


def main() -> None:
    p = argparse.ArgumentParser(description="Run downstream signal test on FNSPID")
    p.add_argument(
        "--source",
        choices=["prefix", "full"],
        default="prefix",
        help="prefix = HTTP-Range partial download; " "full = read pre-downloaded 5.7 GB CSV",
    )
    p.add_argument(
        "--n-bytes", type=int, default=50_000_000, help="prefix mode: partial-download size (bytes)"
    )
    p.add_argument("--date-from", default="2014-01-01")
    p.add_argument("--date-to", default="2020-06-01")
    p.add_argument("--max-rows", type=int, default=0, help="Final cap on rows (0 = no cap)")
    p.add_argument(
        "--max-per-ticker-total",
        type=int,
        default=200,
        help="full mode: hard cap of headlines per ticker over the " "whole date range",
    )
    p.add_argument(
        "--no-per-day-cap",
        action="store_true",
        help="full mode: keep every headline of a (ticker, day) "
        "instead of collapsing to the longest one",
    )
    p.add_argument(
        "--meta-source",
        choices=["twitter", "phrasebank"],
        default="twitter",
        help="cache used to fit the stacking meta-learner",
    )
    p.add_argument(
        "--ticker-whitelist",
        choices=["full", "legacy_ab"],
        default="full",
        help="full = ~362-name S&P-500 (Option C); " "legacy_ab = original 32 A-B-letter slice",
    )
    p.add_argument("--llm-workers", type=int, default=8)
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM agent inference but still keep its "
        "column in the meta-learner (uniform 1/3 placeholder); "
        "use --no-llm to drop the LLM agent entirely",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Drop the LLM agent everywhere: no inference, no "
        "column in the meta-learner, no field in "
        "predictions.json. Used when the LLM provider's "
        "rate limits make a 30k-headline run infeasible.",
    )
    p.add_argument("--out-dir", default=None, help="override results/fnspid sub-dir for this run")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(
        "  Section 5.X: Downstream Signal Test on FNSPID "
        f"(source={args.source}, meta={args.meta_source})"
    )
    print("=" * 70)

    max_rows_arg = args.max_rows if args.max_rows > 0 else None
    whitelist = (
        LEGACY_AB_WHITELIST if args.ticker_whitelist == "legacy_ab" else DEFAULT_TICKER_WHITELIST
    )
    if args.source == "full":
        sl = load_fnspid_full(
            date_from=args.date_from,
            date_to=args.date_to,
            ticker_whitelist=whitelist,
            max_per_ticker_per_day=None if args.no_per_day_cap else 1,
            max_per_ticker_total=args.max_per_ticker_total,
            max_rows=max_rows_arg,
        )
    else:
        sl = load_fnspid_slice(
            n_bytes=args.n_bytes,
            date_from=args.date_from,
            date_to=args.date_to,
            ticker_whitelist=whitelist,
            max_rows=max_rows_arg if max_rows_arg is not None else 4000,
        )
    print(
        f"  [FNSPID] {sl.rows_after_filter} rows | "
        f"{len(sl.tickers)} tickers | {sl.date_min} → {sl.date_max}"
    )
    texts: list[str] = sl.df["text"].tolist()
    pre_texts = preprocess_batch(texts)

    if args.meta_source == "phrasebank":
        ds_name = "warwickai/financial_phrasebank_mirror"
        ds_label = "PhraseBank"
        ft_dir = FT_FINBERT_PB_DIR
    else:
        ds_name = "zeroshot/twitter-financial-news-sentiment"
        ds_label = "Twitter"
        ft_dir = FT_FINBERT_TWITTER_DIR

    print(f"\n  [Agent] TF-IDF + LogReg (re-trained on {ds_label} train split)")
    train_local, _, _ = load_financial_phrasebank(DataConfig(dataset_name=ds_name))
    train_local_x = preprocess_batch(train_local.texts)
    tfidf = TfidfLogRegBaseline()
    tfidf.train(train_local_x, train_local.labels)
    tfidf_proba = _tfidf_proba_aligned(tfidf, pre_texts)
    print(f"    proba shape={tfidf_proba.shape}")

    print(f"\n  [Agent] FinBERT (fine-tuned on {ds_label}, loaded from disk)")
    fb = TransformerBaseline()
    if ft_dir.exists():
        fb.load(str(ft_dir))
        fb._id2label = {int(k): v for k, v in fb.model.config.id2label.items()}
        fb._label2id = {v: k for k, v in fb._id2label.items()}
        print(f"    loaded fine-tuned weights from {ft_dir}")
    else:
        print(f"    [WARN] {ft_dir} missing — falling back to zero-shot FinBERT")
    print(f"    inference device: {fb.device}", flush=True)
    finbert_proba = fb.predict_proba(pre_texts, label_order=LABELS, show_progress=True)
    print(f"    proba shape={finbert_proba.shape}", flush=True)

    if args.no_llm:
        print("\n  [Agent] GPT-4o-mini --- dropped from this run (--no-llm).")
        llm_labels = None
        llm_conf = None
        llm_proba = None
        llm_cost = 0.0
        llm_tokens = 0
    else:
        print("\n  [Agent] GPT-4o-mini (zero-shot, concurrent)")
        if args.skip_llm:
            print("    [SKIP] using a uniform 1/3 distribution as placeholder")
            n = len(pre_texts)
            llm_labels = ["neutral"] * n
            llm_conf = np.full(n, 1.0 / 3.0)
            llm_proba = np.full((n, 3), 1.0 / 3.0)
            llm_cost = 0.0
            llm_tokens = 0
        else:
            llm = SingleLLMAgent(mode="zero_shot")
            cache_path = out_dir / "_llm_cache.json"
            llm_labels, llm_conf = _run_llm_concurrent(
                llm, pre_texts, n_workers=args.llm_workers, cache_path=cache_path
            )
            llm_proba = _llm_to_pseudo_proba(llm_labels, llm_conf)
            llm_cost = float(llm.total_cost_usd)
            llm_tokens = int(llm.total_tokens)
        print(f"    proba shape={llm_proba.shape}, cost=${llm_cost:.4f}")

    print("\n  [Agent] Loughran-McDonald lexicon")
    lex = LoughranMcDonaldAgent()
    lex_proba = lex.predict_proba(pre_texts, label_order=LABELS)
    print(f"    proba shape={lex_proba.shape}")

    print(
        f"\n  [Meta] re-fitting stacking meta-learner from {args.meta_source} cache "
        f"({'3-agent (no LLM)' if args.no_llm else '4-agent'})"
    )
    meta_model, meta_info = _fit_meta_from_cache(meta_source=args.meta_source, drop_llm=args.no_llm)
    train_acc_key = f"train_accuracy_on_{args.meta_source}_val"
    print(
        f"    meta-learner train_acc on {args.meta_source} val = "
        f"{meta_info[train_acc_key]:.4f} "
        f"(n={meta_info['n_train_samples']}, p={meta_info['n_features']}, "
        f"agents={meta_info['agent_order']})"
    )

    if args.no_llm:
        test_X = np.hstack([tfidf_proba, finbert_proba, lex_proba])
    else:
        test_X = np.hstack([tfidf_proba, finbert_proba, llm_proba, lex_proba])
    ens_proba = meta_model.predict_proba(test_X)
    ens_pred = [LABELS[i] for i in np.argmax(ens_proba, axis=1)]

    rows = []
    for i in range(len(texts)):
        row = {
            "text": texts[i],
            "ticker": str(sl.df["ticker"].iloc[i]),
            "date": str(sl.df["date"].iloc[i]),
            "tfidf_proba": tfidf_proba[i].tolist(),
            "finbert_proba": finbert_proba[i].tolist(),
            "lex_proba": lex_proba[i].tolist(),
            "ensemble_proba": ens_proba[i].tolist(),
            "tfidf_pred": LABELS[int(np.argmax(tfidf_proba[i]))],
            "finbert_pred": LABELS[int(np.argmax(finbert_proba[i]))],
            "lex_pred": LABELS[int(np.argmax(lex_proba[i]))],
            "ensemble_pred": ens_pred[i],
        }
        if not args.no_llm:
            row.update(
                {
                    "llm_proba": llm_proba[i].tolist(),
                    "llm_pred": llm_labels[i],
                    "llm_confidence": float(llm_conf[i]),
                }
            )
        rows.append(row)

    out = {
        "meta": {
            "fnspid_slice": {
                "n_bytes": int(sl.bytes_downloaded),
                "rows_raw": int(sl.rows_raw),
                "rows_after_filter": int(sl.rows_after_filter),
                "date_min": str(sl.date_min),
                "date_max": str(sl.date_max),
                "tickers": list(sl.tickers),
                "source_url": getattr(sl, "source_url", None),
                "source_mode": args.source,
                "args": {
                    "date_from": args.date_from,
                    "date_to": args.date_to,
                    "max_rows": args.max_rows,
                    "max_per_ticker_total": args.max_per_ticker_total,
                    "no_per_day_cap": bool(args.no_per_day_cap),
                    "meta_source": args.meta_source,
                    "no_llm": bool(args.no_llm),
                },
            },
            "meta_learner": meta_info,
            "llm_cost_usd": llm_cost,
            "llm_tokens": llm_tokens,
            "label_order": LABELS,
            "agents": (
                ["tfidf", "finbert", "lex", "ensemble"]
                if args.no_llm
                else ["tfidf", "finbert", "llm", "lex", "ensemble"]
            ),
        },
        "rows": rows,
    }

    out_path = out_dir / "predictions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved {len(rows)} rows to {out_path}")
    print("\nPredictions phase complete.")


if __name__ == "__main__":
    main()
