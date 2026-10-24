"""Calibration (risk quantification) : la PD estimée est-elle cohérente avec les défauts observés ?

Tests implémentés, du plus simple au plus réaliste :

- ``binomial_test`` : par grade, H0 « PD vraie ≤ PD estimée ». Suppose des défauts
  indépendants → trop sévère quand les défauts sont corrélés (cycle économique).
- ``jeffreys_test`` : approche bayésienne recommandée par la BCE dans ses
  instructions de reporting (2019) : loi a posteriori Beta(D + ½ ; N − D + ½),
  p-value = F_Beta(PD estimée). Se comporte mieux sur petits effectifs.
- ``vasicek_binomial_test`` : seuil critique ajusté de la corrélation d'actifs
  (facteur systémique) : challenger du test binomial pour l'hypothèse d'indépendance.
- ``hosmer_lemeshow`` : test global du χ² sur l'ensemble des grades.
- ``traffic_light`` : synthèse vert / orange / rouge par grade.

Pour les portefeuilles à faible taux de défaut (LDP, Grande Clientèle), les tests
par grade manquent de puissance : on agrège les grades, on travaille sur plusieurs
années ou on recourt à des bornes de confiance (Pluto & Tasche, 2005).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, chi2, norm


def grade_table(df: pd.DataFrame, grade_col="grade", pd_col="pd",
                target="default_12m") -> pd.DataFrame:
    g = df.groupby(grade_col).agg(N=(target, "size"), D=(target, "sum"), PD=(pd_col, "mean"))
    g["TD_observe"] = g["D"] / g["N"]
    return g.reset_index()


def binomial_test(t: pd.DataFrame) -> pd.Series:
    """p-value unilatérale P(X ≥ D | N, PD) ; faible ⇒ PD sous-estimée."""
    return pd.Series(binom.sf(t["D"] - 1, t["N"], t["PD"]), index=t.index)


def jeffreys_test(t: pd.DataFrame) -> pd.Series:
    """p-value du test de Jeffreys (BCE) : F_Beta(PD ; D + ½, N − D + ½)."""
    return pd.Series(beta.cdf(t["PD"], t["D"] + 0.5, t["N"] - t["D"] + 0.5), index=t.index)


def vasicek_binomial_test(t: pd.DataFrame, rho: float = 0.12,
                          alpha: float = 0.01) -> pd.DataFrame:
    """Taux de défaut critique tenant compte d'une corrélation entre défauts.

    Sous le modèle à un facteur, le taux de défaut d'un grand portefeuille vaut au
    quantile q : N((G(PD) + √ρ·G(q)) / √(1−ρ)). Au-delà, la PD est rejetée."""
    crit = norm.cdf((norm.ppf(t["PD"]) + np.sqrt(rho) * norm.ppf(1 - alpha)) / np.sqrt(1 - rho))
    crit_binom = binom.ppf(1 - alpha, t["N"], t["PD"]) / t["N"]
    return pd.DataFrame({"TD_critique_binomial": crit_binom, "TD_critique_vasicek": crit,
                         "rejet_binomial": t["TD_observe"] > crit_binom,
                         "rejet_vasicek": t["TD_observe"] > crit}, index=t.index)


def hosmer_lemeshow(t: pd.DataFrame, out_of_sample: bool = True) -> dict[str, float]:
    """Statistique HL = Σ (D_k − N_k·p_k)² / (N_k·p_k·(1 − p_k)).
    Degrés de liberté : K hors échantillon de développement, K − 2 sinon."""
    stat = float((((t["D"] - t["N"] * t["PD"]) ** 2) / (t["N"] * t["PD"] * (1 - t["PD"]))).sum())
    dof = len(t) if out_of_sample else len(t) - 2
    return {"statistique": stat, "ddl": dof, "p_value": float(chi2.sf(stat, dof))}


def traffic_light(p_values: pd.Series, amber: float = 0.05, red: float = 0.01) -> pd.Series:
    return pd.cut(p_values, [-np.inf, red, amber, np.inf], labels=["ROUGE", "ORANGE", "VERT"])


def calibration_report(df: pd.DataFrame, rho: float = 0.12, **kw) -> pd.DataFrame:
    t = grade_table(df, **kw)
    t["p_binomial"] = binomial_test(t)
    t["p_jeffreys"] = jeffreys_test(t)
    t["feu_jeffreys"] = traffic_light(t["p_jeffreys"])
    return pd.concat([t, vasicek_binomial_test(t, rho=rho)], axis=1)


def lgd_backtest(df: pd.DataFrame, est="lgd", real="lgd_realised",
                 segment="product") -> pd.DataFrame:
    """Backtesting LGD sur les défauts : test t unilatéral H0 « LGD estimée ≥ LGD réalisée »."""
    from scipy.stats import ttest_1samp

    d = df[df[real].notna()]
    rows = []
    for seg, s in d.groupby(segment):
        diff = s[real] - s[est]
        res = ttest_1samp(diff, 0.0, alternative="greater")
        rows.append({segment: seg, "n_defauts": len(s), "LGD_estimee": s[est].mean(),
                     "LGD_realisee": s[real].mean(), "p_value": res.pvalue})
    return pd.DataFrame(rows)
