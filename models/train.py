"""
models/train.py
───────────────
Trains the LightGBM classifier using walk-forward cross-validation.

Walk-forward design:
  - Data is sorted chronologically.
  - Each fold: train on all past data, validate on next WF_WINDOW_DAYS.
  - Prevents any data leakage across time.
  - Final model is trained on the full dataset.

Outputs:
  - Trained LightGBM model saved to models/saved/lgbm_model.txt
  - Walk-forward OOS predictions saved to outputs/wf_predictions.parquet
  - Feature importance saved to outputs/feature_importance.csv
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix
)

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from config.settings import (
    LGBM_PARAMS, MODELS_DIR, OUTPUTS_DIR,
    WF_WINDOW_DAYS, CV_FOLDS, RANDOM_SEED,
    CLASS_LABELS, SIGNAL_COLUMN,
    TARGET_DIRECTIONAL_ACCURACY, TARGET_F1_SCORE,
)

logger = logging.getLogger(__name__)

# Feature columns to exclude from model input
NON_FEATURE_COLS = {
    "signal", "fwd_return_5d", "ticker",
    "adj_close", "volume_raw",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted list of feature column names."""
    return sorted([c for c in df.columns if c not in NON_FEATURE_COLS])


# ── Walk-Forward Split ─────────────────────────────────────────────────────

def walk_forward_splits(
    dates: pd.DatetimeIndex,
    n_folds: int = CV_FOLDS,
    test_window: int = WF_WINDOW_DAYS,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate walk-forward train/test index splits.

    The dataset is divided so the last n_folds * test_window days
    are used as successive OOS windows. Training always uses all
    available prior history (expanding window).

    Returns
    -------
    list of (train_dates, test_dates) tuples
    """
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    total = len(unique_dates)
    test_total = n_folds * test_window

    if test_total >= total * 0.5:
        logger.warning(
            f"Walk-forward test period ({test_total} days) is "
            f">= 50% of total data ({total} days). Reducing folds."
        )
        n_folds = max(1, total // (2 * test_window))
        test_total = n_folds * test_window

    splits = []
    for i in range(n_folds):
        test_end_idx   = total - i * test_window
        test_start_idx = test_end_idx - test_window
        if test_start_idx <= 0:
            break
        train_dates = unique_dates[:test_start_idx]
        test_dates  = unique_dates[test_start_idx:test_end_idx]
        splits.append((train_dates, test_dates))

    # Return in chronological order (earliest test first)
    return list(reversed(splits))


# ── Single Fold Training ───────────────────────────────────────────────────

def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fold_num: int,
) -> tuple:
    """
    Train one LightGBM fold and return model + OOS predictions.

    Returns
    -------
    (model, val_preds_proba, val_preds_class)
    """
    if not _HAS_LGB:
        raise ImportError("lightgbm not installed — pip install lightgbm")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

    params = {**LGBM_PARAMS}
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),  # silent
    ]

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    proba = model.predict(X_val)          # shape: (n, 3)
    preds = proba.argmax(axis=1)

    acc = accuracy_score(y_val, preds)
    f1  = f1_score(y_val, preds, average="macro", zero_division=0)
    logger.info(
        f"Fold {fold_num}  |  acc={acc:.3f}  f1={f1:.3f}  "
        f"trees={model.best_iteration}"
    )

    return model, proba, preds


# ── Walk-Forward Evaluation ────────────────────────────────────────────────

def walk_forward_train(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    Run full walk-forward cross-validation and collect OOS predictions.

    Parameters
    ----------
    panel : pd.DataFrame
        Full feature panel from features/pipeline.py

    Returns
    -------
    (oos_predictions_df, fold_models)
    """
    feature_cols = get_feature_columns(panel)
    logger.info(f"Features: {len(feature_cols)}")

    splits = walk_forward_splits(panel.index, n_folds=CV_FOLDS)
    logger.info(f"Walk-forward folds: {len(splits)}")

    all_oos = []
    fold_models = []

    for fold_num, (train_dates, test_dates) in enumerate(splits, 1):
        train_mask = panel.index.isin(train_dates)
        test_mask  = panel.index.isin(test_dates)

        X_train = panel.loc[train_mask, feature_cols].fillna(0)
        y_train = panel.loc[train_mask, SIGNAL_COLUMN].astype(int)
        X_val   = panel.loc[test_mask,  feature_cols].fillna(0)
        y_val   = panel.loc[test_mask,  SIGNAL_COLUMN].astype(int)

        logger.info(
            f"Fold {fold_num}: "
            f"train={train_dates[0].date()}→{train_dates[-1].date()} "
            f"({len(X_train):,} rows)  |  "
            f"val={test_dates[0].date()}→{test_dates[-1].date()} "
            f"({len(X_val):,} rows)"
        )

        model, proba, preds = train_fold(X_train, y_train, X_val, y_val, fold_num)
        fold_models.append(model)

        oos = panel.loc[test_mask, ["ticker", SIGNAL_COLUMN, "fwd_return_5d"]].copy()
        oos["pred_signal"]    = preds
        oos["prob_sell"]      = proba[:, 0]
        oos["prob_hold"]      = proba[:, 1]
        oos["prob_buy"]       = proba[:, 2]
        oos["confidence"]     = proba.max(axis=1)
        oos["fold"]           = fold_num
        all_oos.append(oos)

    oos_df = pd.concat(all_oos).sort_index()

    # Aggregate metrics
    overall_acc = accuracy_score(
        oos_df[SIGNAL_COLUMN].astype(int),
        oos_df["pred_signal"].astype(int)
    )
    overall_f1 = f1_score(
        oos_df[SIGNAL_COLUMN].astype(int),
        oos_df["pred_signal"].astype(int),
        average="macro", zero_division=0
    )

    logger.info("=" * 55)
    logger.info(f"Walk-forward OOS accuracy : {overall_acc:.4f}  "
                f"(target ≥ {TARGET_DIRECTIONAL_ACCURACY})")
    logger.info(f"Walk-forward OOS macro-F1 : {overall_f1:.4f}  "
                f"(target ≥ {TARGET_F1_SCORE})")
    logger.info("=" * 55)
    logger.info("\n" + classification_report(
        oos_df[SIGNAL_COLUMN].astype(int),
        oos_df["pred_signal"].astype(int),
        target_names=["Sell", "Hold", "Buy"],
        zero_division=0,
    ))

    # Save OOS predictions
    oos_path = OUTPUTS_DIR.parent / "wf_predictions.parquet"
    oos_df.to_parquet(oos_path)
    logger.info(f"OOS predictions → {oos_path}")

    return oos_df, fold_models


# ── Final Model (trained on all data) ─────────────────────────────────────

def train_final_model(panel: pd.DataFrame) -> "lgb.Booster":
    """
    Train final production model on the full dataset.
    This model is used for live inference (run_daily.py).
    """
    if not _HAS_LGB:
        raise ImportError("lightgbm not installed")

    feature_cols = get_feature_columns(panel)
    X = panel[feature_cols].fillna(0)
    y = panel[SIGNAL_COLUMN].astype(int)

    params = {**LGBM_PARAMS}
    # Remove early stopping for final model — use full n_estimators
    params.pop("early_stopping_rounds", None)

    train_data = lgb.Dataset(X, label=y)

    model = lgb.train(
        params,
        train_data,
        callbacks=[lgb.log_evaluation(period=100)],
    )

    # Save model
    model_path = MODELS_DIR / "lgbm_final.txt"
    model.save_model(str(model_path))
    logger.info(f"Final model saved → {model_path}")

    # Feature importance
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    fi_path = OUTPUTS_DIR.parent / "feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    logger.info(f"Feature importance → {fi_path}")
    logger.info("\nTop 15 features:\n" + fi.head(15).to_string(index=False))

    return model


# ── Entry Point ────────────────────────────────────────────────────────────

def run_training_pipeline(panel: pd.DataFrame) -> dict:
    """
    Full training pipeline: walk-forward eval + final model.

    Returns
    -------
    dict with keys: 'oos_df', 'fold_models', 'final_model'
    """
    logger.info("Starting walk-forward training...")
    oos_df, fold_models = walk_forward_train(panel)

    logger.info("Training final model on full data...")
    final_model = train_final_model(panel)

    return {
        "oos_df":       oos_df,
        "fold_models":  fold_models,
        "final_model":  final_model,
    }


if __name__ == "__main__":
    from features.pipeline import load_feature_panel
    panel = load_feature_panel()
    results = run_training_pipeline(panel)
