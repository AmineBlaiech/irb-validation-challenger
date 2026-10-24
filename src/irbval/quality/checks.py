"""Contrôles de qualité des données d'entrée des modèles PD / LGD / CCF.

Les contrôles sont organisés selon les dimensions de qualité de données citées
par le superviseur (guide BCE des modèles internes, section « Data governance »,
et principes BCBS 239) : complétude, exactitude/validité, cohérence, unicité,
actualité/représentativité.

Chaque règle renvoie un masque booléen des lignes en anomalie. Le rapport donne,
pour chaque règle, le nombre et le taux d'anomalies et un statut vert/orange/rouge
selon des seuils paramétrables. Les contrôles statistiques (valeurs extrêmes,
mécanisme des valeurs manquantes, fuite de données) sont des méthodes
« challengers » : ils ne décident pas, ils signalent ce qu'un validateur doit examiner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


class Dimension(StrEnum):
    COMPLETENESS = "Complétude"
    VALIDITY = "Validité"
    CONSISTENCY = "Cohérence"
    UNIQUENESS = "Unicité"
    ACCURACY = "Exactitude (valeurs extrêmes)"
    REPRESENTATIVENESS = "Représentativité"


type RuleFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(slots=True)
class Rule:
    name: str
    dimension: Dimension
    description: str
    check: RuleFn
    amber: float = 0.001  # taux d'anomalie au-delà duquel le statut passe à orange
    red: float = 0.01  # … puis à rouge


@dataclass
class DataQualityReport:
    results: pd.DataFrame
    flags: pd.DataFrame = field(repr=False)

    def summary(self) -> pd.DataFrame:
        return self.results.groupby(["dimension", "status"]).size().unstack(fill_value=0)


def _status(rate: float, amber: float, red: float) -> str:
    return "ROUGE" if rate >= red else "ORANGE" if rate >= amber else "VERT"


def run_rules(df: pd.DataFrame, rules: list[Rule]) -> DataQualityReport:
    rows, flags = [], {}
    for r in rules:
        mask = r.check(df).fillna(False).astype(bool)
        n_fail = int(mask.sum())
        rate = n_fail / len(df) if len(df) else 0.0
        flags[r.name] = mask
        rows.append({"rule": r.name, "dimension": str(r.dimension), "description": r.description,
                     "n_anomalies": n_fail, "taux": rate, "status": _status(rate, r.amber, r.red)})
    return DataQualityReport(pd.DataFrame(rows), pd.DataFrame(flags))


# --------------------------------------------------------------------------------------
# Détection de valeurs extrêmes : trois méthodes challengers aux propriétés différentes
# --------------------------------------------------------------------------------------

def outliers_iqr(x: pd.Series, k: float = 3.0) -> pd.Series:
    """Règle de Tukey : hors de [Q1 − k·IQR ; Q3 + k·IQR]. Robuste, mais suppose une
    distribution à peu près symétrique ; sur un revenu (asymétrique) elle sur-détecte
    à droite → préférer l'appliquer sur log(x)."""
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def outliers_mad(x: pd.Series, threshold: float = 5.0) -> pd.Series:
    """Score z robuste fondé sur la médiane et le MAD (|x − med| / (1,4826·MAD)).
    Insensible aux valeurs extrêmes elles-mêmes, contrairement au z-score classique."""
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=x.index)
    return ((x - med).abs() / mad) > threshold


def outliers_percentile(x: pd.Series, lower: float = 0.001, upper: float = 0.999) -> pd.Series:
    """Hors des percentiles [0,1 % ; 99,9 %] : simple et non paramétrique, mais détecte
    toujours ~0,2 % de la population même si aucune valeur n'est erronée."""
    lo, hi = x.quantile([lower, upper])
    return (x < lo) | (x > hi)


# --------------------------------------------------------------------------------------
# Analyses statistiques complémentaires
# --------------------------------------------------------------------------------------

def missingness_vs_target(df: pd.DataFrame, target: str = "default_12m") -> pd.DataFrame:
    """Teste si le fait d'être manquant est lié au défaut (χ² d'indépendance).

    Si oui, les valeurs ne sont pas manquantes « complètement au hasard » (MCAR) :
    supprimer les lignes ou imputer par la médiane biaise le modèle, et la modalité
    « manquant » porte elle-même de l'information de risque.
    """
    rows = []
    for col in df.columns:
        if col == target or col.startswith("_"):
            continue
        miss = df[col].isna()
        if miss.sum() == 0 or miss.all():
            continue
        table = pd.crosstab(miss, df[target])
        _, p_value, _, _ = chi2_contingency(table)
        rows.append({
            "variable": col,
            "taux_manquant": miss.mean(),
            "TD_si_manquant": df.loc[miss, target].mean(),
            "TD_si_renseigné": df.loc[~miss, target].mean(),
            "p_value_chi2": p_value,
            "conclusion": "Non MCAR : manquant informatif" if p_value < 0.01 else "Compatible MCAR",
        })
    return pd.DataFrame(rows)


def leakage_scan(df: pd.DataFrame, target: str = "default_12m",
                 threshold: float = 0.90) -> pd.DataFrame:
    """Repère les variables « trop belles » : AUC univariée > seuil.

    Une variable seule qui sépare presque parfaitement les défauts contient souvent
    une information postérieure à la date d'observation (fuite de données)."""
    from irbval.performance.discrimination import auc

    rows = []
    y = df[target].to_numpy()
    for col in df.select_dtypes("number").columns:
        if col in {target, "pd", "lgd_realised", "cohort"} or col.startswith("_"):
            continue
        x = df[col]
        ok = x.notna().to_numpy()
        if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
            continue
        a = auc(y[ok], x.to_numpy()[ok])
        a = max(a, 1 - a)
        rows.append({"variable": col, "auc_univariee": a, "suspect_fuite": a >= threshold})
    return pd.DataFrame(rows).sort_values("auc_univariee", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------------------
# Jeu de règles par défaut pour le portefeuille Retail simulé
# --------------------------------------------------------------------------------------

def default_rules() -> list[Rule]:
    def missing(col: str) -> RuleFn:
        return lambda d: d[col].isna()

    return [
        Rule("revenu_manquant", Dimension.COMPLETENESS, "Revenu non renseigné", missing("income"),
             amber=0.01, red=0.05),
        Rule("grade_manquant", Dimension.COMPLETENESS, "Note interne absente", missing("grade")),
        Rule("ead_manquante", Dimension.COMPLETENESS, "EAD absente", missing("ead")),
        Rule("doublons", Dimension.UNIQUENESS, "Même contrat à la même date d'arrêté",
             lambda d: d.duplicated(["contract_id", "snapshot_date"], keep="first")),
        Rule("age_invalide", Dimension.VALIDITY, "Âge hors [18 ; 110]",
             lambda d: ~d["age"].between(18, 110)),
        Rule("ead_negative", Dimension.VALIDITY, "EAD < 0", lambda d: d["ead"] < 0),
        Rule("pd_hors_bornes", Dimension.VALIDITY, "PD hors ]0 ; 1]",
             lambda d: ~((d["pd"] > 0) & (d["pd"] <= 1))),
        Rule("lgd_realisee_hors_bornes", Dimension.VALIDITY, "LGD réalisée hors [0 ; 1]",
             lambda d: d["lgd_realised"].notna() & ~d["lgd_realised"].between(0, 1)),
        Rule("tire_superieur_limite", Dimension.CONSISTENCY, "Encours tiré > 110 % de la limite",
             lambda d: d["drawn_amount"] > 1.1 * d["credit_limit"]),
        Rule("octroi_apres_arrete", Dimension.CONSISTENCY, "Date d'octroi > date d'arrêté",
             lambda d: d["origination_date"] > d["snapshot_date"]),
        Rule("defaut_sans_date", Dimension.CONSISTENCY, "Défaut = 1 mais date de défaut absente",
             lambda d: (d["default_12m"] == 1) & d["default_date"].isna()),
        Rule("date_defaut_sans_defaut", Dimension.CONSISTENCY, "Date de défaut sans indicateur",
             lambda d: (d["default_12m"] == 0) & d["default_date"].notna()),
        Rule("pd_incoherente_grade", Dimension.CONSISTENCY,
             "PD différente de la PD de l'échelle maîtresse pour ce grade",
             lambda d: d.groupby("grade")["pd"].transform("nunique") > 1),
        Rule("revenu_extreme_mad", Dimension.ACCURACY, "log(revenu) : score z robuste > 5",
             lambda d: outliers_mad(np.log(d["income"].clip(lower=1)))),
    ]
