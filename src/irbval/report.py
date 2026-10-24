"""Pipeline de validation challenger : enchaîne les contrôles et produit un rapport Markdown."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from irbval.performance import calibration as cal  # noqa: E402
from irbval.performance import discrimination as disc  # noqa: E402
from irbval.performance import stability as stab  # noqa: E402
from irbval.quality import checks as dq  # noqa: E402
from irbval.rwa import irb  # noqa: E402

PRODUCT_TO_CLASS = {
    "mortgage": irb.ExposureClass.RETAIL_MORTGAGE,
    "consumer": irb.ExposureClass.RETAIL_OTHER,
    "revolving": irb.ExposureClass.RETAIL_QRRE,
}
# Pondérations standard simplifiées, illustratives, pour le plancher de sortie
ILLUSTRATIVE_SA_RW = {"mortgage": 0.35, "consumer": 0.75, "revolving": 0.75}


def _md(df: pd.DataFrame, floatfmt: str = ".4f", fmts: dict[str, str] | None = None) -> str:
    fmts = fmts or {}

    def fmt(v, col):
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else format(v, fmts.get(col, floatfmt))
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v, c) for v, c in zip(row, df.columns, strict=True)) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def clean(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Traitement minimal documenté : retrait des doublons et des lignes invalides
    bloquantes. Les manquants de revenu sont conservés (information de risque)."""
    blocking = ["doublons", "ead_negative", "octroi_apres_arrete"]
    keep = ~flags[blocking].any(axis=1)
    out = df.loc[keep.to_numpy()].copy()
    out.loc[~out["age"].between(18, 110), "age"] = np.nan
    out["lgd_realised"] = out["lgd_realised"].clip(0, 1)
    return out


def run(df: pd.DataFrame, out_dir: str | Path = "reports") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sections: list[str] = ["# Rapport de validation challenger — portefeuille Retail simulé\n",
                           "> Données entièrement simulées. Objectif : démontrer une démarche de "
                           "validation indépendante, pas juger un modèle réel.\n"]

    # 1. Qualité des données ------------------------------------------------------------
    rep = dq.run_rules(df, dq.default_rules())
    sections += ["## 1. Qualité des données\n", _md(rep.results.drop(columns="description")), ""]
    if "_anomaly" in df:  # mesure de l'efficacité des contrôles sur les anomalies injectées
        truth = df["_anomaly"].str.contains("duplicate|age_999|negative_ead|origination_after")
        found = rep.flags[["doublons", "age_invalide", "ead_negative", "octroi_apres_arrete"]]
        recall = (found.any(axis=1) & truth).sum() / max(truth.sum(), 1)
        sections.append(f"Taux de détection des anomalies injectées (règles déterministes) : "
                        f"**{recall:.1%}**\n")
    miss = dq.missingness_vs_target(df.drop(columns=["default_date", "lgd_realised"]))
    sections += ["### Mécanisme des valeurs manquantes\n", _md(miss), ""]
    leak = dq.leakage_scan(df).head(5)
    sections += ["### Détection de fuite de données (AUC univariée)\n", _md(leak), ""]
    inc = df["income"].dropna()
    comp = pd.DataFrame([
        {"méthode": "IQR (k=3) sur revenu brut", "n_detectes": int(dq.outliers_iqr(inc).sum())},
        {"méthode": "IQR (k=3) sur log(revenu)",
         "n_detectes": int(dq.outliers_iqr(np.log(inc)).sum())},
        {"méthode": "MAD (z robuste > 5) sur log(revenu)",
         "n_detectes": int(dq.outliers_mad(np.log(inc)).sum())},
        {"méthode": "Percentiles 0,1 % / 99,9 %",
         "n_detectes": int(dq.outliers_percentile(inc).sum())},
    ])
    n_true = int(df["_anomaly"].str.contains("income_x100").sum()) if "_anomaly" in df else 0
    sections += [f"### Comparaison des méthodes de détection des valeurs extrêmes "
                 f"(revenus erronés injectés : {n_true})\n", _md(comp), ""]

    data = clean(df, rep.flags)
    dev, cur = data[data["cohort"] == 2023], data[data["cohort"] == 2025]

    # 2. Discrimination ------------------------------------------------------------------
    a_dev, lo, hi = disc.auc_confidence_interval(dev["default_12m"], dev["pd"])
    b_lo, b_hi = disc.auc_bootstrap_ci(cur["default_12m"].to_numpy(), cur["pd"].to_numpy(),
                                       n_boot=200)
    t = disc.auc_degradation_test(a_dev, cur["default_12m"], cur["pd"])
    sections += [
        "## 2. Pouvoir discriminant\n",
        f"- AUC développement : **{a_dev:.3f}** (IC 95 % DeLong [{lo:.3f} ; {hi:.3f}]) — "
        f"Gini {2 * a_dev - 1:.3f}",
        f"- AUC suivi 2025 : **{t['auc_courante']:.3f}** (IC 95 % bootstrap "
        f"[{b_lo:.3f} ; {b_hi:.3f}])",
        f"- Test de dégradation (AUC initiale vs courante) : statistique "
        f"{t['statistique']:.2f}, p-value {t['p_value']:.3f}\n",
    ]
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, s in [("Développement 2023", dev), ("Suivi 2025", cur)]:
        fpr, tpr = disc.roc_curve(s["default_12m"], s["pd"])
        ax.plot(fpr, tpr, label=f"{name} (AUC {disc.auc(s['default_12m'], s['pd']):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbes ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=130)
    plt.close(fig)
    sections.append("![ROC](roc.png)\n")

    # 3. Calibration ---------------------------------------------------------------------
    calib = cal.calibration_report(cur, rho=0.03)
    hl = cal.hosmer_lemeshow(cal.grade_table(cur))
    cols = ["grade", "N", "D", "PD", "TD_observe", "p_binomial", "p_jeffreys", "feu_jeffreys",
            "rejet_binomial", "rejet_vasicek"]
    pct = {"PD": ".2%", "TD_observe": ".2%"}
    sections += ["## 3. Calibration (cohorte 2025)\n", _md(calib[cols], fmts=pct), "",
                 f"Hosmer-Lemeshow : χ² = {hl['statistique']:.1f}, ddl = {hl['ddl']}, "
                 f"p-value = {hl['p_value']:.2e}\n",
                 "Lecture : 2025 est simulée comme une année modérément dégradée ET les grades "
                 "G08 à G10 sont volontairement sous-estimés de 30 %. Le test binomial (défauts "
                 "indépendants) et le test de Jeffreys rejettent ces grades ; le test ajusté de "
                 "la corrélation (Vasicek, ρ = 3 %, grand portefeuille) est beaucoup plus "
                 "tolérant et ne rejette rien. Aucun test seul ne sépare effet de cycle et "
                 "mauvaise calibration : il faut confronter la PD à la moyenne long terme des "
                 "taux de défaut sur plusieurs années.\n"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(calib))
    ax.bar(x, calib["TD_observe"], color="#9db4c0", label="Taux de défaut observé")
    ax.plot(x, calib["PD"], "o-", color="#1b4965", label="PD estimée")
    ax.set_yscale("log")
    ax.set_xticks(x, calib["grade"])
    ax.set(title="PD estimée vs taux de défaut observé (2025)", ylabel="échelle log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=130)
    plt.close(fig)
    sections += ["![Calibration](calibration.png)\n",
                 "### Backtesting LGD\n", _md(cal.lgd_backtest(cur)), ""]

    # 4. Stabilité -------------------------------------------------------------------------
    psis = stab.psi_table(dev, cur, ["grade", "age", "income", "debt_ratio", "seniority_years"])
    h = stab.herfindahl(cur["grade"])
    sections += ["## 4. Stabilité et représentativité\n", _md(psis), "",
                 f"Concentration de l'échelle (HHI normalisé, 2025) : {h['hhi_normalise']:.4f}\n"]

    # 5. RWA --------------------------------------------------------------------------------
    perf = cur.assign(exposure_class=cur["product"].map(PRODUCT_TO_CLASS))
    perf = perf[perf["ead"] > 0]
    res3 = irb.compute_portfolio_rwa(perf, irb.Regime.crr3())
    res2 = irb.compute_portfolio_rwa(perf, irb.Regime.crr2())
    by = res3.groupby("product").agg(EAD=("ead", "sum"), RWA=("rwa", "sum"), EL=("el", "sum"))
    by["RW_moyen"] = by["RWA"] / by["EAD"]
    by["RWA_CRR2"] = res2.groupby("product")["rwa"].sum()
    rwa_sa = float((perf["ead"] * perf["product"].map(ILLUSTRATIVE_SA_RW)).sum())
    years = (2025, 2027, 2030)
    floors = pd.DataFrame([irb.output_floor(by["RWA"].sum(), rwa_sa, y) for y in years])
    floors.insert(0, "année", [2025, 2027, 2030])
    sens = irb.rwa_sensitivity(0.02, 0.40, "retail_other")
    sections += ["## 5. Calculateur RWA (IRB, CRR3)\n",
                 _md(by.reset_index(), ",.0f", {"RW_moyen": ".1%"}), "",
                 "Le passage de CRR2 à CRR3 supprime le facteur 1,06 et relève les planchers "
                 "de PD (0,03 % → 0,05 %, 0,10 % pour les revolving) ; des planchers de LGD "
                 "s'ajoutent en A-IRB.\n",
                 "### Plancher de sortie (RWA standard illustratifs)\n",
                 _md(floors, ",.0f", {"floor_pct": ".1%", "binding": ".0f"}), "",
                 "### Sensibilité du RW à la PD et à la LGD (PD 2 %, LGD 40 %, autre retail)\n",
                 _md(sens), ""]

    path = out / "rapport_validation.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path
