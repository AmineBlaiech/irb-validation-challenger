"""Portefeuille Retail simulé, avec anomalies de qualité de données injectées volontairement.

Aucune donnée réelle n'est utilisée. Le générateur produit :
- une échelle maîtresse de 10 grades + défaut, avec une PD par grade ;
- deux cohortes : échantillon de développement (2023) et de suivi/backtesting (2025),
  avec une légère dérive de population et un facteur systémique annuel (Vasicek)
  qui crée des défauts corrélés — ce qui rend le test binomial « naïf » trop sévère ;
- des variables d'entrée (revenu, âge, taux d'endettement, ancienneté…) et des
  paramètres LGD / EAD / CCF ;
- des anomalies connues (valeurs manquantes non aléatoires, doublons, valeurs
  aberrantes, incohérences de dates, LGD hors bornes, fuite de données) afin que
  les contrôles puissent être évalués : on sait ce qu'ils doivent détecter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Échelle maîtresse illustrative (PD à 1 an par grade)
MASTER_SCALE = pd.DataFrame({
    "grade": [f"G{i:02d}" for i in range(1, 11)],
    "pd_grade": [0.0005, 0.0012, 0.0025, 0.0045, 0.0080, 0.0140, 0.0250, 0.0450, 0.0900, 0.2000],
})


def _simulate_cohort(rng: np.random.Generator, n: int, year: int, drift: float,
                     systemic_z: float, rho: float = 0.10, underestimation: float = 0.0,
                     debt_shift: float = 0.0) -> pd.DataFrame:
    """Simule une cohorte d'emprunteurs particuliers (crédit à la consommation / habitat)."""
    score = rng.normal(drift, 1.0, n)  # score latent : plus élevé = plus risqué
    cuts = np.quantile(rng.normal(0, 1, 200_000), np.linspace(0, 1, 11)[1:-1])
    grade_idx = np.searchsorted(cuts, score)
    pd_est = MASTER_SCALE["pd_grade"].to_numpy()[grade_idx]

    # PD « vraie » : égale à la PD estimée, sauf sous-estimation volontaire des grades G08-G10
    true_pd = np.clip(pd_est * (1 + underestimation * (grade_idx >= 7)), 1e-5, 0.99)
    # Défauts corrélés via un facteur systémique (modèle de Vasicek à un facteur)
    cond_pd = norm.cdf((norm.ppf(true_pd) - np.sqrt(rho) * systemic_z) / np.sqrt(1 - rho))
    default = rng.random(n) < cond_pd

    age = np.clip(rng.normal(45, 13, n) - 3 * score, 18, 95).round()
    income = np.exp(rng.normal(10.3 - 0.12 * score, 0.45, n)).round(-1)
    debt_ratio = np.clip(rng.beta(2, 5, n) + 0.05 * score + debt_shift, 0, 0.95)
    seniority = np.clip(rng.exponential(6, n) - score, 0, 40).round(1)
    product = rng.choice(["mortgage", "consumer", "revolving"], n, p=[0.5, 0.35, 0.15])
    limit = np.where(product == "mortgage", rng.uniform(80e3, 400e3, n),
                     np.where(product == "consumer", rng.uniform(3e3, 40e3, n),
                              rng.uniform(1e3, 8e3, n))).round(-2)
    drawn = (limit * np.where(product == "revolving", rng.uniform(0.1, 0.9, n),
                              rng.uniform(0.6, 1.0, n))).round(2)
    ccf_est = np.where(product == "revolving", 0.65, 0.0)
    ead = drawn + ccf_est * (limit - drawn)
    lgd_est = np.where(product == "mortgage", 0.15, np.where(product == "consumer", 0.40, 0.55))
    lgd_real = np.where(default, np.clip(lgd_est + rng.normal(0.02, 0.15, n), 0, 1), np.nan)

    snapshot = pd.Timestamp(f"{year}-12-31")
    origination = snapshot - pd.to_timedelta(rng.integers(30, 3650, n), unit="D")
    default_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    days = pd.to_timedelta(rng.integers(1, 365, int(default.sum())), unit="D")
    default_date[default] = snapshot + days

    return pd.DataFrame({
        "contract_id": [f"C{year}{i:06d}" for i in range(n)],
        "snapshot_date": snapshot,
        "cohort": year,
        "product": product,
        "origination_date": origination,
        "age": age,
        "income": income,
        "debt_ratio": debt_ratio.round(4),
        "seniority_years": seniority,
        "credit_limit": limit,
        "drawn_amount": drawn,
        "ead": ead.round(2),
        "grade": MASTER_SCALE["grade"].to_numpy()[grade_idx],
        "pd": pd_est,
        "lgd": lgd_est,
        "ccf": ccf_est,
        "default_12m": default.astype(int),
        "default_date": default_date,
        "lgd_realised": lgd_real,
    })


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Injecte des anomalies connues ; leurs positions sont tracées dans ``_anomaly``."""
    df = df.copy()
    df["_anomaly"] = ""
    n = len(df)

    def tag(mask, label):
        df.loc[mask, "_anomaly"] = df.loc[mask, "_anomaly"] + label + ";"

    # 1. Revenu manquant NON aléatoirement (MAR) : plus fréquent chez les futurs défauts
    p_miss = np.where(df["default_12m"] == 1, 0.12, 0.02)
    m = rng.random(n) < p_miss
    df.loc[m, "income"] = np.nan
    tag(m, "missing_income")
    # 2. Âge aberrant (valeur « bouche-trou » 999)
    m = rng.random(n) < 0.002
    df.loc[m, "age"] = 999
    tag(m, "age_999")
    # 3. Revenus extrêmes (erreur d'unité : saisie en centimes)
    m = rng.random(n) < 0.001
    df.loc[m, "income"] = df.loc[m, "income"] * 100
    tag(m, "income_x100")
    # 4. EAD négative
    m = rng.random(n) < 0.0005
    df.loc[m, "ead"] = -df.loc[m, "ead"]
    tag(m, "negative_ead")
    # 5. Encours tiré > limite autorisée de plus de 10 %
    m = rng.random(n) < 0.002
    df.loc[m, "drawn_amount"] = df.loc[m, "credit_limit"] * 1.5
    tag(m, "drawn_gt_limit")
    # 6. Date d'octroi postérieure à la date d'arrêté
    m = rng.random(n) < 0.001
    df.loc[m, "origination_date"] = df.loc[m, "snapshot_date"] + pd.Timedelta(days=40)
    tag(m, "origination_after_snapshot")
    # 7. Défaut sans date de défaut
    m = (df["default_12m"] == 1) & (rng.random(n) < 0.03)
    df.loc[m, "default_date"] = pd.NaT
    tag(m, "default_without_date")
    # 8. LGD réalisée hors de [0, 1] (frais de recouvrement non plafonnés)
    m = df["lgd_realised"].notna() & (rng.random(n) < 0.02)
    df.loc[m, "lgd_realised"] = 1.25
    tag(m, "lgd_out_of_bounds")
    # 9. Doublons exacts d'enregistrements
    dup_idx = rng.choice(df.index, size=int(0.003 * n), replace=False)
    dups = df.loc[dup_idx].copy()
    dups["_anomaly"] = dups["_anomaly"] + "duplicate;"
    df = pd.concat([df, dups], ignore_index=True)
    # 10. Variable de fuite : nombre d'impayés observés APRÈS la date d'arrêté
    df["nb_unpaid_next_12m"] = np.where(df["default_12m"] == 1,
                                        rng.integers(2, 6, len(df)), rng.integers(0, 2, len(df)))
    return df


def simulate_portfolio(n_dev: int = 40_000, n_mon: int = 30_000,
                       seed: int = 42) -> pd.DataFrame:
    """Retourne un portefeuille (développement 2023 + suivi 2025) avec anomalies injectées."""
    rng = np.random.default_rng(seed)
    dev = _simulate_cohort(rng, n_dev, 2023, drift=0.0, systemic_z=0.0)
    # 2025 : année modérément dégradée (choc systémique), grades G08-G10 sous-estimés de 30 %
    # et hausse des taux d'endettement (+6 points) : la « vérité terrain » est connue.
    mon = _simulate_cohort(rng, n_mon, 2025, drift=0.15, systemic_z=-0.3,
                           underestimation=0.30, debt_shift=0.06)
    return _inject_anomalies(pd.concat([dev, mon], ignore_index=True), rng)
