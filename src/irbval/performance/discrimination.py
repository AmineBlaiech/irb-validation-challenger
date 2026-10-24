"""Pouvoir discriminant (risk differentiation) : AUC, Gini, intervalles et tests.

- AUC calculée via la statistique de Mann-Whitney (avec gestion des ex-æquo,
  indispensable sur une note discrète à 10 grades).
- Variance de l'AUC par la méthode de DeLong (1988).
- Test de dégradation de l'AUC entre le développement et la période courante,
  dans l'esprit des instructions BCE de reporting des résultats de validation (2019) :
  S = (AUC_init − AUC_courante) / σ_courante, p-value = 1 − Φ(S).
- Intervalle de confiance bootstrap, challenger non paramétrique de DeLong.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """AUC = P(score_défaut > score_sain) + 0,5·P(égalité). Score croissant avec le risque."""
    y = np.asarray(y).astype(bool)
    ranks = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gini(y, score) -> float:
    return 2 * auc(y, score) - 1


def delong_variance(y: np.ndarray, score: np.ndarray) -> float:
    """Variance de l'AUC (DeLong, DeLong & Clarke-Pearson, 1988), via les rangs."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    pos, neg = s[y], s[~y]
    m, n = len(pos), len(neg)
    r_all = rankdata(np.concatenate([pos, neg]))
    r_pos, r_neg = rankdata(pos), rankdata(neg)
    v10 = (r_all[:m] - r_pos) / n  # composantes structurelles des défauts
    v01 = 1 - (r_all[m:] - r_neg) / m  # … et des sains
    return float(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n)


def auc_confidence_interval(y, score, alpha: float = 0.05) -> tuple[float, float, float]:
    a = auc(y, score)
    se = np.sqrt(delong_variance(y, score))
    z = norm.ppf(1 - alpha / 2)
    return a, max(0.0, a - z * se), min(1.0, a + z * se)


def auc_bootstrap_ci(y, score, n_boot: int = 500, alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            stats.append(auc(y[idx], score[idx]))
    return tuple(np.quantile(stats, [alpha / 2, 1 - alpha / 2]))


def auc_degradation_test(auc_initial: float, y_current, score_current) -> dict[str, float]:
    """Test unilatéral : l'AUC courante est-elle significativement inférieure à l'AUC initiale ?"""
    a_cur = auc(y_current, score_current)
    s = np.sqrt(delong_variance(y_current, score_current))
    stat = (auc_initial - a_cur) / s
    return {"auc_initiale": auc_initial, "auc_courante": a_cur, "ecart_type": s,
            "statistique": stat, "p_value": float(1 - norm.cdf(stat))}


def roc_curve(y, score) -> tuple[np.ndarray, np.ndarray]:
    """Points (taux de faux positifs, taux de vrais positifs) par seuil décroissant."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    thresholds = np.unique(s)[::-1]
    tpr = np.array([0.0] + [(s[y] >= t).mean() for t in thresholds])
    fpr = np.array([0.0] + [(s[~y] >= t).mean() for t in thresholds])
    return fpr, tpr
