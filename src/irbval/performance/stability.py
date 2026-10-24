"""Stabilité et représentativité : la population actuelle ressemble-t-elle à celle
du développement ?"""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index = Σ (a − e)·ln(a / e).

    Variables continues découpées sur les déciles de l'échantillon de référence.
    Lecture usuelle : < 0,10 stable ; 0,10–0,25 à surveiller ; > 0,25 dérive significative."""
    e, a = expected.dropna(), actual.dropna()
    if e.dtype.kind in "fi" and e.nunique() > bins:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e_dist = pd.cut(e, edges).value_counts(normalize=True, sort=False)
        a_dist = pd.cut(a, edges).value_counts(normalize=True, sort=False)
    else:
        e_dist = e.value_counts(normalize=True)
        a_dist = a.value_counts(normalize=True).reindex(e_dist.index, fill_value=0)
    e_dist, a_dist = e_dist.clip(lower=1e-6), a_dist.clip(lower=1e-6)
    return float(((a_dist - e_dist) * np.log(a_dist / e_dist)).sum())


def psi_table(dev: pd.DataFrame, cur: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = [{"variable": c, "psi": psi(dev[c], cur[c])} for c in columns]
    out = pd.DataFrame(rows)
    out["lecture"] = pd.cut(out["psi"], [-np.inf, 0.10, 0.25, np.inf],
                            labels=["Stable", "À surveiller", "Dérive"])
    return out


def herfindahl(grades: pd.Series) -> dict[str, float]:
    """Indice de Herfindahl-Hirschman des effectifs par grade (concentration de l'échelle).
    HHI normalisé ∈ [0 ; 1] : 0 = répartition uniforme, 1 = un seul grade."""
    s = grades.value_counts(normalize=True)
    k = len(s)
    hhi = float((s**2).sum())
    return {"hhi": hhi, "hhi_normalise": (hhi - 1 / k) / (1 - 1 / k) if k > 1 else 1.0}


def migration_matrix(prev: pd.Series, curr: pd.Series) -> pd.DataFrame:
    """Matrice de transition des grades (lignes : grade initial ; colonnes : grade final)."""
    return pd.crosstab(prev, curr, normalize="index")


def migration_stability(matrix: pd.DataFrame) -> dict[str, float]:
    """Part des contrats restés dans leur grade (diagonale) ou à ±1 grade."""
    m = matrix.to_numpy()
    k = min(m.shape)
    diag = np.mean([m[i, i] for i in range(k)])
    near = np.mean([m[i, max(0, i - 1): i + 2].sum() for i in range(k)])
    return {"part_diagonale": float(diag), "part_plus_moins_1": float(near)}
