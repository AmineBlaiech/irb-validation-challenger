"""Calculateur d'exigences en fonds propres – approche IRB (CRR, art. 153-154 ; CRR3).

Formule ASRF (Asymptotic Single Risk Factor, modèle de Vasicek) :

    K = [LGD · N( (G(PD) + √R · G(0,999)) / √(1−R) ) − PD · LGD] · MA
    RWA = K · 12,5 · EAD

- N : fonction de répartition de la loi normale centrée réduite, G = N⁻¹.
- R : corrélation d'actifs (fixée par le régulateur selon la classe d'exposition).
- MA : ajustement de maturité (expositions non retail uniquement).

Sous CRR3 (règlement (UE) 2024/1623, applicable depuis le 1er janvier 2025) :
- le facteur scalaire 1,06 disparaît de la formule (paramètre ``scaling_factor``) ;
- des planchers de paramètres (« input floors ») s'appliquent : PD ≥ 0,05 %
  (0,10 % pour les revolving QRRE), planchers de LGD en A-IRB ;
- un plancher de sortie (« output floor ») limite le gain de l'IRB à x % des RWA
  standard, x passant de 50 % (2025) à 72,5 % (2030).

Les chiffres sont ceux des textes publics ; ils doivent être relus contre le CRR
consolidé avant tout usage réglementaire.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.stats import norm

CONFIDENCE = 0.999


class ExposureClass(StrEnum):
    CORPORATE = "corporate"
    CORPORATE_SME = "corporate_sme"
    RETAIL_MORTGAGE = "retail_mortgage"
    RETAIL_QRRE = "retail_qrre"
    RETAIL_QRRE_TRANSACTOR = "retail_qrre_transactor"
    RETAIL_OTHER = "retail_other"


RETAIL_CLASSES = {
    ExposureClass.RETAIL_MORTGAGE,
    ExposureClass.RETAIL_QRRE,
    ExposureClass.RETAIL_QRRE_TRANSACTOR,
    ExposureClass.RETAIL_OTHER,
}


@dataclass(frozen=True, slots=True)
class Regime:
    """Paramètres réglementaires du calcul (CRR2 ou CRR3)."""

    name: str
    scaling_factor: float
    pd_floor: dict[ExposureClass, float]
    apply_lgd_floors: bool

    @classmethod
    def crr2(cls) -> Regime:
        return cls(
            name="CRR2",
            scaling_factor=1.06,
            pd_floor={c: 0.0003 for c in ExposureClass},
            apply_lgd_floors=False,
        )

    @classmethod
    def crr3(cls) -> Regime:
        floors = {c: 0.0005 for c in ExposureClass}
        floors[ExposureClass.RETAIL_QRRE] = 0.0010  # revolvers QRRE
        return cls(name="CRR3", scaling_factor=1.0, pd_floor=floors, apply_lgd_floors=True)


# Planchers de LGD A-IRB (CRR3) – sélection pédagogique, à vérifier contre le texte.
LGD_FLOORS_CRR3: dict[ExposureClass, float] = {
    ExposureClass.CORPORATE: 0.25,  # non sécurisé
    ExposureClass.CORPORATE_SME: 0.25,
    ExposureClass.RETAIL_MORTGAGE: 0.05,  # garanti par un bien résidentiel
    ExposureClass.RETAIL_QRRE: 0.50,
    ExposureClass.RETAIL_QRRE_TRANSACTOR: 0.50,
    ExposureClass.RETAIL_OTHER: 0.30,  # non sécurisé
}

# Calendrier du plancher de sortie (CRR3, art. 465)
OUTPUT_FLOOR_SCHEDULE: dict[int, float] = {
    2025: 0.50, 2026: 0.55, 2027: 0.60, 2028: 0.65, 2029: 0.70, 2030: 0.725,
}


def asset_correlation(pd_: np.ndarray, exposure_class: ExposureClass,
                      turnover_meur: np.ndarray | None = None) -> np.ndarray:
    """Corrélation d'actifs réglementaire R selon la classe d'exposition."""
    pd_ = np.asarray(pd_, dtype=float)
    match exposure_class:
        case ExposureClass.RETAIL_MORTGAGE:
            return np.full_like(pd_, 0.15)
        case ExposureClass.RETAIL_QRRE | ExposureClass.RETAIL_QRRE_TRANSACTOR:
            return np.full_like(pd_, 0.04)
        case ExposureClass.RETAIL_OTHER:
            w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
            return 0.03 * w + 0.16 * (1 - w)
        case ExposureClass.CORPORATE | ExposureClass.CORPORATE_SME:
            w = (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
            r = 0.12 * w + 0.24 * (1 - w)
            if exposure_class is ExposureClass.CORPORATE_SME:
                if turnover_meur is None:
                    raise ValueError("Le chiffre d'affaires (M€) est requis pour l'ajustement PME.")
                s = np.clip(np.asarray(turnover_meur, dtype=float), 5, 50)
                r = r - 0.04 * (1 - (s - 5) / 45)
            return r
    raise ValueError(f"Classe d'exposition inconnue : {exposure_class}")


def maturity_adjustment(pd_: np.ndarray, maturity: np.ndarray) -> np.ndarray:
    """Ajustement de maturité MA = (1 + (M − 2,5)·b) / (1 − 1,5·b),
    avec b = (0,11852 − 0,05478·ln PD)²."""
    b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
    m = np.clip(maturity, 1, 5)
    return (1 + (m - 2.5) * b) / (1 - 1.5 * b)


def capital_requirement(pd_, lgd, exposure_class: ExposureClass | str, maturity=2.5,
                        turnover_meur=None, regime: Regime | None = None,
                        elbe=None) -> np.ndarray:
    """Exigence de fonds propres K (en % de l'EAD) pour des expositions IRB.

    - PD = 1 (défaut) : K = max(0, LGD − ELBE) sans facteur d'échelle.
    - Planchers de PD et LGD appliqués selon ``regime`` (CRR3 par défaut).
    """
    regime = regime or Regime.crr3()
    cls = ExposureClass(exposure_class)
    pd_ = np.atleast_1d(np.asarray(pd_, dtype=float))
    lgd = np.broadcast_to(np.asarray(lgd, dtype=float), pd_.shape).copy()
    defaulted = pd_ >= 1.0

    pd_eff = np.maximum(pd_, regime.pd_floor[cls])
    if regime.apply_lgd_floors:
        lgd = np.where(defaulted, lgd, np.maximum(lgd, LGD_FLOORS_CRR3[cls]))

    pd_nd = np.where(defaulted, 0.5, pd_eff)  # valeur neutre pour éviter les NaN
    r = asset_correlation(pd_nd, cls, turnover_meur)
    cond_pd = norm.cdf((norm.ppf(pd_nd) + np.sqrt(r) * norm.ppf(CONFIDENCE)) / np.sqrt(1 - r))
    k = lgd * cond_pd - pd_nd * lgd
    if cls not in RETAIL_CLASSES:
        k = k * maturity_adjustment(pd_nd, np.broadcast_to(maturity, pd_.shape))
    k = k * regime.scaling_factor

    elbe_arr = np.zeros_like(pd_) if elbe is None else np.broadcast_to(elbe, pd_.shape)
    k_def = np.maximum(0.0, lgd - elbe_arr)
    return np.where(defaulted, k_def, k)


def risk_weight(pd_, lgd, exposure_class, **kwargs) -> np.ndarray:
    """Pondération en risque RW = 12,5 · K."""
    return 12.5 * capital_requirement(pd_, lgd, exposure_class, **kwargs)


def compute_portfolio_rwa(df: pd.DataFrame, regime: Regime | None = None, *,
                          pd_col="pd", lgd_col="lgd", ead_col="ead",
                          class_col="exposure_class", maturity_col: str | None = None,
                          turnover_col: str | None = None) -> pd.DataFrame:
    """Calcule K, RW, RWA et la perte attendue (EL = PD·LGD·EAD) ligne à ligne."""
    regime = regime or Regime.crr3()
    out = df.copy()
    k = pd.Series(np.nan, index=df.index)
    for cls, idx in df.groupby(class_col).groups.items():
        sub = df.loc[idx]
        k.loc[idx] = capital_requirement(
            sub[pd_col].to_numpy(), sub[lgd_col].to_numpy(), cls,
            maturity=sub[maturity_col].to_numpy() if maturity_col else 2.5,
            turnover_meur=sub[turnover_col].to_numpy() if turnover_col else None,
            regime=regime,
        )
    out["k"] = k
    out["rw"] = 12.5 * out["k"]
    out["rwa"] = out["rw"] * out[ead_col]
    out["el"] = np.minimum(out[pd_col], 1.0) * out[lgd_col] * out[ead_col]
    return out


def output_floor(rwa_irb: float, rwa_sa: float, year: int = 2030) -> dict[str, float]:
    """Plancher de sortie : RWA retenus = max(RWA IRB, x · RWA standard)."""
    x = OUTPUT_FLOOR_SCHEDULE[min(max(year, 2025), 2030)]
    floored = max(rwa_irb, x * rwa_sa)
    return {"floor_pct": x, "rwa_irb": rwa_irb, "rwa_floor": x * rwa_sa,
            "rwa_final": floored, "binding": float(floored > rwa_irb)}


def rwa_sensitivity(pd_: float, lgd: float, exposure_class, shocks=(-0.2, -0.1, 0.1, 0.2),
                    **kwargs) -> pd.DataFrame:
    """Sensibilité du RW à des chocs relatifs sur la PD et la LGD (outil de challenge)."""
    base = float(risk_weight(pd_, lgd, exposure_class, **kwargs)[0])
    rows = []
    for s in shocks:
        rw_pd = float(risk_weight(pd_ * (1 + s), lgd, exposure_class, **kwargs)[0])
        rw_lgd = float(risk_weight(pd_, lgd * (1 + s), exposure_class, **kwargs)[0])
        rows.append({"choc": f"{s:+.0%}", "rw_choc_pd": rw_pd, "rw_choc_lgd": rw_lgd,
                     "delta_rw_pd": rw_pd / base - 1, "delta_rw_lgd": rw_lgd / base - 1})
    return pd.DataFrame(rows)
