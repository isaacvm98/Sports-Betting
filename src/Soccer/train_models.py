"""
Production training pipeline for the live soccer draw module.

Produces the two artifacts the live scanner consumes:
  - Data/soccer_models/draw_xgb.pkl     (XGBoost classifier, target: was_draw)
  - Data/soccer_models/cox_model.pkl    (Cox PH survival model for equalize timing)

Both pkls are written with the exact schema draw_scanner.py expects
(see _load_xgb_model / _load_cox_model). A sidecar JSON next to each
pkl records the git SHA, dataset hash, hyperparameters, and OOS metrics
for that specific artifact — so we can audit what's running in prod.

Usage:
    # Dry run — train, evaluate, print metrics. Nothing written.
    python -m src.Soccer.train_models

    # Same but save (with backup of the previous artifact).
    python -m src.Soccer.train_models --save

    # Custom seasons:
    python -m src.Soccer.train_models \\
        --train-season 2024/2025 --eval-season 2025/2026 --save

    # Skip data-quality gate (NOT recommended for live deploys):
    python -m src.Soccer.train_models --save --force

Real-money safety guardrails:
  - Default is dry-run; --save is opt-in.
  - Pre-existing pkls are backed up to Data/soccer_models/_archive/
    with a timestamp before being overwritten.
  - Atomic writes: pickle to .tmp, then os.replace().
  - Data-quality gate refuses to save if >5%% of qualifying matches have
    truncated goals arrays (use --force to bypass; see project memory
    for context on the matches.json corruption issue).
  - Post-save smoke test: re-instantiate DrawScanner and verify it
    loads the new artifacts without errors.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import xgboost as xgb

from src.Soccer.feature_builder import (
    LEAGUES,
    MATCHES_FILE as DEFAULT_MATCHES_FILE,
    build_features,
    get_feature_columns,
    _get_season,
)
from src.Soccer.survival_model import build_survival_data, train_cox

logger = logging.getLogger("train_models")

DEFAULT_MODEL_DIR = Path("Data/soccer_models")

# Default to the same hyperparameters that produced the current live model.
XGB_HYPERPARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

DATA_QUALITY_THRESHOLD = 0.05  # max acceptable fraction of truncated goals arrays


# ───────────────────────── data quality gate ─────────────────────────


def assess_matches_json(matches: List[Dict]) -> Dict:
    """Sanity-check the dataset before we train on it.

    Checks:
      1. was_draw field vs goals-array final score (parser sanity).
      2. Whether `cards` field is present and red rate is plausible.
      3. Whether `substitutions` field is present.
    """
    n = len(matches)
    disagreements = 0
    n_with_was = 0
    n_with_cards = 0
    n_with_subs = 0
    total_red_cards = 0
    total_subs = 0

    for m in matches:
        was = m.get("was_draw")
        if was is not None:
            n_with_was += 1
            goals = m.get("goals", []) or []
            if goals:
                final_h = goals[-1]["home_after"]
                final_a = goals[-1]["away_after"]
                from_goals = (final_h == final_a)
            else:
                from_goals = True
            if bool(was) != from_goals:
                disagreements += 1

        if "cards" in m:
            n_with_cards += 1
            for c in m["cards"] or []:
                if c.get("card_type") in ("red", "second_yellow"):
                    total_red_cards += 1
        if "substitutions" in m:
            n_with_subs += 1
            total_subs += len(m["substitutions"] or [])

    rate = disagreements / n_with_was if n_with_was else 0.0
    return {
        "n_matches": n,
        "n_with_was_draw": n_with_was,
        "n_disagreements": disagreements,
        "disagreement_rate": rate,
        "n_with_cards_field": n_with_cards,
        "n_with_subs_field": n_with_subs,
        "red_cards_per_match": (
            total_red_cards / n_with_cards if n_with_cards else 0.0
        ),
        "subs_per_match": (
            total_subs / n_with_subs if n_with_subs else 0.0
        ),
    }


# ───────────────────────── training ─────────────────────────


def train_xgb(df: pd.DataFrame, train_season: str, feature_cols: List[str]):
    # Only drop rows missing the *essential* features (xg, score state).
    # Card/sub features are intentionally NaN for matches with no red
    # card or no subs yet — XGBoost handles those natively. Dropping
    # them here would discard ~78% of training rows.
    from src.Soccer.feature_builder import CARD_FEATURE_COLS, SUB_FEATURE_COLS
    optional = set(CARD_FEATURE_COLS) | set(SUB_FEATURE_COLS)
    essential = [c for c in feature_cols if c in df.columns and c not in optional]
    df_train = df[df["season"] == train_season].dropna(subset=essential)
    if df_train.empty:
        raise RuntimeError(f"No training rows for season {train_season}")

    X_tr = df_train[feature_cols].values
    y_tr = df_train["was_draw"].values

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    scale = n_neg / n_pos if n_pos > 0 else 1.0

    clf = xgb.XGBClassifier(scale_pos_weight=scale, **XGB_HYPERPARAMS)
    clf.fit(X_tr, y_tr, verbose=False)
    cal = CalibratedClassifierCV(clf, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)

    return cal, clf, df_train


def evaluate_xgb(cal, df: pd.DataFrame, eval_season: str, feature_cols: List[str]) -> Dict:
    from src.Soccer.feature_builder import CARD_FEATURE_COLS, SUB_FEATURE_COLS
    optional = set(CARD_FEATURE_COLS) | set(SUB_FEATURE_COLS)
    essential = [c for c in feature_cols if c in df.columns and c not in optional]
    df_eval = df[df["season"] == eval_season].dropna(subset=essential)
    if df_eval.empty:
        return {"n_eval": 0}

    X = df_eval[feature_cols].values
    y = df_eval["was_draw"].values
    probs = cal.predict_proba(X)[:, 1]

    metrics: Dict = {
        "n_eval": int(len(df_eval)),
        "n_draws": int(y.sum()),
        "actual_draw_rate": float(y.mean()),
        "mean_pred": float(probs.mean()),
        "calibration_gap": float(probs.mean() - y.mean()),
    }
    if len(np.unique(y)) > 1:
        metrics.update({
            "auc": float(roc_auc_score(y, probs)),
            "brier": float(brier_score_loss(y, probs)),
            "log_loss": float(log_loss(y, probs)),
        })
    return metrics


# ───────────────────────── safe write ─────────────────────────


def _atomic_pickle(obj, path: Path):
    """Write to .tmp then os.replace — never leave a half-written pkl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _backup_existing(path: Path, archive_dir: Path) -> Optional[Path]:
    if not path.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = archive_dir / f"{path.stem}.{ts}{path.suffix}"
    dest.write_bytes(path.read_bytes())
    return dest


# ───────────────────────── provenance ─────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True
        ).strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ───────────────────────── smoke test ─────────────────────────


def smoke_test_loaders(xgb_path: Path, cox_path: Path):
    """Replicate the scanner's loader contract on the freshly-written files.

    Catches schema drift before the new pkls reach the live trader.
    The scanner reads `state["calibrated"]` and `state["feature_cols"]`
    for XGB, and pickle.load() the raw CoxPHFitter for Cox — see
    src/Soccer/draw_scanner.py:_load_xgb_model / _load_cox_model.
    """
    with open(xgb_path, "rb") as f:
        state = pickle.load(f)
    cal = state["calibrated"]
    feat_cols = state["feature_cols"]
    if cal is None or not feat_cols:
        raise RuntimeError("XGB pkl loaded but missing calibrated/feature_cols")

    X = np.zeros((1, len(feat_cols)))
    p = float(cal.predict_proba(X)[0, 1])
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"XGB returned out-of-range probability: {p}")

    if cox_path.exists():
        with open(cox_path, "rb") as f:
            cox = pickle.load(f)
        if not hasattr(cox, "predict_survival_function"):
            raise RuntimeError("Cox pkl is not a fitted CoxPHFitter")

    return {"xgb_features": len(feat_cols), "xgb_test_pred": p}


# ───────────────────────── main ─────────────────────────


def run(
    train_season: str,
    eval_season: Optional[str],
    save: bool,
    force: bool,
    matches_file: Path,
    out_dir: Path,
    with_cards: bool,
    with_subs: bool,
):
    xgb_path = out_dir / "draw_xgb.pkl"
    cox_path = out_dir / "cox_model.pkl"
    archive_dir = out_dir / "_archive"
    sidecar_path = out_dir / "draw_xgb.meta.json"

    print("=" * 78)
    print("  SOCCER DRAW MODEL — TRAINING PIPELINE")
    print("=" * 78)
    print(f"  Train season: {train_season}")
    print(f"  Eval season:  {eval_season or '(none)'}")
    print(f"  Matches file: {matches_file}")
    print(f"  Output dir:   {out_dir}")
    print(f"  With cards:   {with_cards}")
    print(f"  With subs:    {with_subs}")
    print(f"  Save:         {save}  (force={force})")
    print()

    # 1. Load + assess data
    with open(matches_file, "r", encoding="utf-8") as f:
        matches = json.load(f).get("matches", [])
    quality = assess_matches_json(matches)
    print(f"  matches.json rows: {quality['n_matches']}  "
          f"(was_draw set on {quality['n_with_was_draw']})")
    print(f"  was_draw vs goals-array disagreements: "
          f"{quality['n_disagreements']} "
          f"({quality['disagreement_rate']*100:.1f}%)")
    print(f"  cards field present:  {quality['n_with_cards_field']}/"
          f"{quality['n_matches']}  "
          f"(red rate {quality['red_cards_per_match']:.2f}/match)")
    print(f"  subs field present:   {quality['n_with_subs_field']}/"
          f"{quality['n_matches']}  "
          f"(avg {quality['subs_per_match']:.1f} subs/match)")

    if with_cards and quality["n_with_cards_field"] < quality["n_matches"] * 0.95:
        print()
        print(f"  REFUSING TO TRAIN with --with-cards: only "
              f"{quality['n_with_cards_field']}/{quality['n_matches']} "
              f"matches have a `cards` field. Re-scrape first or drop "
              f"--with-cards.")
        if not force:
            sys.exit(2)
    if with_subs and quality["n_with_subs_field"] < quality["n_matches"] * 0.95:
        print()
        print(f"  REFUSING TO TRAIN with --with-subs: only "
              f"{quality['n_with_subs_field']}/{quality['n_matches']} "
              f"matches have a `substitutions` field.")
        if not force:
            sys.exit(2)

    if quality["disagreement_rate"] > DATA_QUALITY_THRESHOLD and save and not force:
        print()
        print(f"  REFUSING TO SAVE: disagreement rate "
              f"{quality['disagreement_rate']*100:.1f}% exceeds threshold "
              f"{DATA_QUALITY_THRESHOLD*100:.1f}%.")
        print(f"  matches.json has truncated goals arrays. Re-scrape, or "
              f"pass --force to override (NOT recommended for live deploy).")
        sys.exit(2)
    elif quality["disagreement_rate"] > DATA_QUALITY_THRESHOLD:
        print(f"  WARNING: disagreement rate above threshold; saving anyway "
              f"because {'force=True' if force else 'save=False'}.")

    # 2. Build features (no momentum, with team stats — same as live model)
    df = build_features(matches)
    match_lookup = {str(m["match_id"]): m for m in matches}
    df["was_draw"] = df["match_id"].apply(
        lambda mid: int(match_lookup.get(str(mid), {}).get("was_draw", False))
    )
    df["season"] = df["date"].apply(_get_season)

    feature_cols = get_feature_columns(
        include_momentum=False,
        include_team=True,
        include_cards=with_cards,
        include_subs=with_subs,
    )
    feature_cols = [c for c in feature_cols if c != "momentum"]

    print(f"\n  Feature cols ({len(feature_cols)}): {feature_cols}")
    seasons_in_df = sorted(df["season"].unique())
    print(f"  Seasons available: {seasons_in_df}")
    if train_season not in seasons_in_df:
        raise RuntimeError(f"Train season {train_season} not in dataset.")

    # 3. Train XGBoost
    print(f"\n  Training XGBoost on {train_season} ...")
    cal, _raw, df_train = train_xgb(df, train_season, feature_cols)
    print(f"    n_train={len(df_train)}  draws={int(df_train['was_draw'].sum())} "
          f"({df_train['was_draw'].mean()*100:.1f}%)")

    # 4. Train Cox
    print(f"\n  Training Cox PH on {train_season} ...")
    surv_df = build_survival_data(matches)
    surv_train = surv_df[surv_df["season"] == train_season]
    if surv_train.empty:
        raise RuntimeError(f"No Cox training rows for {train_season}")
    cox = train_cox(surv_train)
    print(f"    n_obs={len(surv_train)}  C-index={cox.concordance_index_:.3f}")

    # 5. OOS evaluation
    metrics = {}
    if eval_season:
        print(f"\n  OOS evaluation on {eval_season} ...")
        metrics = evaluate_xgb(cal, df, eval_season, feature_cols)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:18s}: {v:.4f}")
            else:
                print(f"    {k:18s}: {v}")

    # 6. Save
    if not save:
        print("\n  [DRY RUN] not writing artifacts. Pass --save to persist.")
        return

    print("\n  Saving artifacts ...")
    matches_sha = _file_sha256(matches_file)
    git_sha = _git_sha()
    now = datetime.now(timezone.utc).isoformat()

    xgb_state = {
        "calibrated": cal,
        "feature_cols": feature_cols,
        "trained_at": now,
        "train_season": train_season,
        "n_train": int(len(df_train)),
        "n_draws_train": int(df_train["was_draw"].sum()),
        "hyperparams": XGB_HYPERPARAMS,
        "git_sha": git_sha,
        "matches_file": str(matches_file),
        "matches_sha256": matches_sha,
    }
    cox_meta = {
        "trained_at": now,
        "train_season": train_season,
        "n_obs": int(len(surv_train)),
        "concordance_index": float(cox.concordance_index_),
        "git_sha": git_sha,
        "matches_file": str(matches_file),
        "matches_sha256": matches_sha,
    }

    bk_xgb = _backup_existing(xgb_path, archive_dir)
    bk_cox = _backup_existing(cox_path, archive_dir)
    if bk_xgb:
        print(f"    backed up: {bk_xgb}")
    if bk_cox:
        print(f"    backed up: {bk_cox}")

    _atomic_pickle(xgb_state, xgb_path)
    # Scanner loads cox via raw pickle.load(), not via dict — preserve that contract.
    _atomic_pickle(cox, cox_path)

    sidecar = {
        "xgb": xgb_state | {"calibrated": "<pickled CalibratedClassifierCV>"},
        "cox": cox_meta,
        "eval_season": eval_season,
        "eval_metrics": metrics,
        "data_quality": quality,
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    print(f"    wrote: {xgb_path} ({xgb_path.stat().st_size:,} bytes)")
    print(f"    wrote: {cox_path} ({cox_path.stat().st_size:,} bytes)")
    print(f"    wrote: {sidecar_path}")

    # 7. Smoke test the live loader contract on the freshly-written files
    print("\n  Smoke-testing scanner loader contract ...")
    smoke = smoke_test_loaders(xgb_path, cox_path)
    print(f"    XGB loaded with {smoke['xgb_features']} features; "
          f"test prediction = {smoke['xgb_test_pred']:.4f}")
    print("    Cox loaded and is a fitted CoxPHFitter")
    print("\n  DONE.")


def main():
    parser = argparse.ArgumentParser(description="Train soccer draw models")
    parser.add_argument("--train-season", default="2024/2025",
                        help="Season string used for training (default 2024/2025)")
    parser.add_argument("--eval-season", default="2025/2026",
                        help="Season string used for OOS evaluation. "
                             "Use '' to skip.")
    parser.add_argument("--save", action="store_true",
                        help="Persist new artifacts to --out-dir. "
                             "Default is dry-run.")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the data-quality gate when saving.")
    parser.add_argument("--matches-file", default=str(DEFAULT_MATCHES_FILE),
                        help="Path to matches.json snapshot. Defaults to "
                             "the canonical file.")
    parser.add_argument("--out-dir", default=str(DEFAULT_MODEL_DIR),
                        help="Directory to write draw_xgb.pkl, cox_model.pkl, "
                             "and the sidecar JSON. Use a sandbox dir to "
                             "test the save path without touching live "
                             "artifacts.")
    parser.add_argument("--with-cards", action="store_true",
                        help="Include card-derived features. Requires the "
                             "matches file to have a `cards` field on most "
                             "matches.")
    parser.add_argument("--with-subs", action="store_true",
                        help="Include substitution-derived features. "
                             "Requires `substitutions` field.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)

    run(
        train_season=args.train_season,
        eval_season=args.eval_season or None,
        save=args.save,
        force=args.force,
        matches_file=Path(args.matches_file),
        out_dir=Path(args.out_dir),
        with_cards=args.with_cards,
        with_subs=args.with_subs,
    )


if __name__ == "__main__":
    main()
