"""End-to-end modeling pipeline: sample → train → evaluate → report."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    GOLD_TABLE,
    GOLD_YEAR,
    IMAGES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SAMPLE_N,
    TARGET_COL,
)
from src.evaluation.interpretability import (  # noqa: E402
    logistic_coefficients,
    permutation_importances,
    save_importance_table,
    shap_values_tree,
    tree_importances,
)
from src.evaluation.leakage import compare_random_vs_grouped  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    metrics_frame,
    predict_scores,
    recall_oriented_threshold,
    save_metrics,
)
from src.evaluation.report import write_modelagem_report  # noqa: E402
from src.evaluation.strategic import municipal_risk_table, save_risk_tables  # noqa: E402
from src.modeling.split import grouped_holdout  # noqa: E402
from src.modeling.train import (  # noqa: E402
    fit_candidate,
    pick_champion,
    save_pipeline,
    save_search_report,
)
from src.preprocessing.features import build_model_frame, feature_lists, xy_groups  # noqa: E402
from src.preprocessing.load_s3 import load_gold_sample_cached  # noqa: E402
from src.visualization.model_plots import (  # noqa: E402
    plot_calibration,
    plot_confusion,
    plot_importance,
    plot_leakage_compare,
    plot_learning_curve,
    plot_metric_bars,
    plot_pr,
    plot_roc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_modeling")

CANDIDATES = ("dummy", "logistic", "tree", "random_forest", "knn")


def _thresholded_metrics(y_true, scores) -> dict:
    base = classification_metrics(y_true, scores, threshold=0.5)
    thr = recall_oriented_threshold(y_true, scores, min_precision=0.62)
    rec = classification_metrics(y_true, scores, threshold=thr)
    rec = {f"recall_oriented_{k}": v for k, v in rec.items()}
    return {**base, **rec}


def run() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading sample n=%s year=%s table=%s", SAMPLE_N, GOLD_YEAR, GOLD_TABLE)
    raw = load_gold_sample_cached(table=GOLD_TABLE, n=SAMPLE_N, year=GOLD_YEAR, random_state=RANDOM_STATE)
    frame = build_model_frame(raw)
    X, y, groups = xy_groups(frame)
    numeric, low, high = feature_lists(frame)
    logger.info(
        "frame=%s features=%s (num=%s low=%s high=%s) municipios=%s",
        frame.shape,
        X.shape[1],
        len(numeric),
        len(low),
        len(high),
        groups.nunique(),
    )

    logger.info("Leakage comparison (logistic, default params)")
    leakage = compare_random_vs_grouped(X, y, groups)
    logger.info("leakage: %s", leakage)
    plot_leakage_compare(leakage["random_split_roc_auc"], leakage["grouped_split_roc_auc"])

    Xtr, Xte, ytr, yte, gtr, gte = grouped_holdout(X, y, groups)
    logger.info(
        "holdout grouped: train=%s test=%s mun_train=%s mun_test=%s overlap=%s",
        len(Xtr),
        len(Xte),
        gtr.nunique(),
        gte.nunique(),
        len(set(gtr) & set(gte)),
    )

    results = []
    holdout_rows = []
    holdout_auc = {}
    for name in CANDIDATES:
        logger.info("=== %s ===", name)
        fitted = fit_candidate(name, Xtr, ytr, gtr, numeric, low, high)
        est = fitted["estimator"]
        scores = predict_scores(est, Xte)
        metrics = _thresholded_metrics(yte, scores)
        fitted["holdout"] = metrics
        holdout_auc[name] = metrics["roc_auc"]
        holdout_rows.append({"model": name, **{k: metrics[k] for k in ("roc_auc", "pr_auc", "f1", "recall", "precision", "brier", "cv_roc_auc") if k in metrics}})
        holdout_rows[-1]["cv_roc_auc"] = fitted.get("cv_roc_auc")
        results.append(fitted)
        save_pipeline(est, name)
        logger.info("%s holdout ROC AUC=%.4f cv=%.4f", name, metrics["roc_auc"], fitted.get("cv_roc_auc") or float("nan"))

    compare = metrics_frame(holdout_rows)
    champion = pick_champion(results, holdout_auc)
    champ_fit = next(r for r in results if r["name"] == champion)
    champ = champ_fit["estimator"]
    save_pipeline(champ, "best_model")
    save_search_report(results)
    logger.info("champion=%s", champion)

    scores = predict_scores(champ, Xte)
    thr = recall_oriented_threshold(yte, scores)
    y_pred = (scores >= 0.5).astype(int)
    y_pred_rec = (scores >= thr).astype(int)

    plot_confusion(yte, y_pred, path=IMAGES_DIR / "model_confusion.png")
    plot_confusion(yte, y_pred_rec, title="Matriz de confusão (threshold recall)", path=IMAGES_DIR / "model_confusion_recall.png")
    plot_roc(yte, scores)
    plot_pr(yte, scores)
    plot_calibration(yte, scores)
    plot_metric_bars(compare)

    perm = permutation_importances(champ, Xte, yte)
    save_importance_table(perm, "permutation_importance")
    plot_importance(perm, "Permutation importance (ROC AUC)", path=IMAGES_DIR / "model_permutation_importance.png")

    tree_imp = tree_importances(champ)
    if tree_imp is not None:
        save_importance_table(tree_imp, "tree_importance")
        plot_importance(tree_imp, "Importância (impureza)", path=IMAGES_DIR / "model_tree_importance.png")

    coefs = logistic_coefficients(champ)
    if coefs is None:
        log_fit = next((r for r in results if r["name"] == "logistic"), None)
        if log_fit:
            coefs = logistic_coefficients(log_fit["estimator"])
    if coefs is not None:
        save_importance_table(coefs.head(40), "logistic_coefficients")
        plot_importance(coefs, "Regressão logística |coef|", path=IMAGES_DIR / "model_logistic_coefs.png")

    rf_fit = next((r for r in results if r["name"] == "random_forest"), None)
    if rf_fit is not None:
        try:
            import matplotlib.pyplot as plt
            import shap

            values, Xt, names, _ = shap_values_tree(rf_fit["estimator"], Xte)
            shap.summary_plot(values, Xt, feature_names=list(names), show=False, max_display=18)
            plt.tight_layout()
            plt.savefig(IMAGES_DIR / "model_shap_summary.png", dpi=130, bbox_inches="tight")
            plt.close()
        except Exception:
            logger.exception("SHAP summary skipped")

    learn_X = Xtr.sample(n=min(25000, len(Xtr)), random_state=RANDOM_STATE)
    learn_y = ytr.loc[learn_X.index]
    plot_learning_curve(champ, learn_X, learn_y)

    # Re-score holdout rows and attach municipality context for ranking
    scored_test = Xte.copy()
    scored_test[TARGET_COL] = yte.to_numpy()
    scored_test["y_score"] = scores
    scored_test["id_municipio"] = gte.to_numpy()
    for col in ("sigla_uf", "nome_regiao", "rede", "meta_alfabetizacao_2024", "meta_alfabetizacao_2025", "lag1_taxa_alfabetizacao", "nome_municipio"):
        if col in frame.columns and col not in scored_test.columns:
            # first value per municipality from the modeling frame
            mapping = frame.groupby("id_municipio")[col].first()
            scored_test[col] = scored_test["id_municipio"].map(mapping)

    risk = municipal_risk_table(
        scored_test,
        scored_test[TARGET_COL],
        scored_test["y_score"],
    )
    save_risk_tables(risk)

    champ_metrics = {**champ_fit["holdout"], "champion": champion}
    save_metrics(
        {
            "champion": champion,
            "holdout": holdout_rows,
            "leakage": leakage,
            "champion_metrics": champ_metrics,
            "best_params": champ_fit.get("best_params"),
        }
    )
    write_modelagem_report(
        compare=compare,
        champion=champion,
        champion_metrics=champ_metrics,
        leakage=leakage,
        best_params=champ_fit.get("best_params") or {},
        n_rows=len(frame),
        n_features=X.shape[1],
        n_municipios_train=int(gtr.nunique()),
        n_municipios_test=int(gte.nunique()),
    )
    logger.info("done. reports in %s images in %s", REPORTS_DIR, IMAGES_DIR)


if __name__ == "__main__":
    run()
