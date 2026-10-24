# Rapport de validation challenger — portefeuille Retail simulé

> Données entièrement simulées. Objectif : démontrer une démarche de validation indépendante, pas juger un modèle réel.

## 1. Qualité des données

| rule | dimension | n_anomalies | taux | status |
|---|---|---|---|---|
| revenu_manquant | Complétude | 1709 | 0.0243 | ORANGE |
| grade_manquant | Complétude | 0 | 0.0000 | VERT |
| ead_manquante | Complétude | 0 | 0.0000 | VERT |
| doublons | Unicité | 210 | 0.0030 | ORANGE |
| age_invalide | Validité | 135 | 0.0019 | ORANGE |
| ead_negative | Validité | 27 | 0.0004 | VERT |
| pd_hors_bornes | Validité | 0 | 0.0000 | VERT |
| lgd_realisee_hors_bornes | Validité | 83 | 0.0012 | ORANGE |
| tire_superieur_limite | Cohérence | 143 | 0.0020 | ORANGE |
| octroi_apres_arrete | Cohérence | 84 | 0.0012 | ORANGE |
| defaut_sans_date | Cohérence | 106 | 0.0015 | ORANGE |
| date_defaut_sans_defaut | Cohérence | 0 | 0.0000 | VERT |
| pd_incoherente_grade | Cohérence | 0 | 0.0000 | VERT |
| revenu_extreme_mad | Exactitude (valeurs extrêmes) | 74 | 0.0011 | ORANGE |

Taux de détection des anomalies injectées (règles déterministes) : **100.0%**

### Mécanisme des valeurs manquantes

| variable | taux_manquant | TD_si_manquant | TD_si_renseigné | p_value_chi2 | conclusion |
|---|---|---|---|---|---|
| income | 0.0243 | 0.2229 | 0.0421 | 0.0000 | Non MCAR : manquant informatif |

### Détection de fuite de données (AUC univariée)

| variable | auc_univariee | suspect_fuite |
|---|---|---|
| nb_unpaid_next_12m | 1.0000 | True |
| debt_ratio | 0.6327 | False |
| income | 0.6004 | False |
| seniority_years | 0.5841 | False |
| age | 0.5828 | False |

### Comparaison des méthodes de détection des valeurs extrêmes (revenus erronés injectés : 75)

| méthode | n_detectes |
|---|---|
| IQR (k=3) sur revenu brut | 423 |
| IQR (k=3) sur log(revenu) | 74 |
| MAD (z robuste > 5) sur log(revenu) | 74 |
| Percentiles 0,1 % / 99,9 % | 138 |

## 2. Pouvoir discriminant

- AUC développement : **0.871** (IC 95 % DeLong [0.864 ; 0.878]) — Gini 0.742
- AUC suivi 2025 : **0.867** (IC 95 % bootstrap [0.860 ; 0.873])
- Test de dégradation (AUC initiale vs courante) : statistique 1.13, p-value 0.128

![ROC](roc.png)

## 3. Calibration (cohorte 2025)

| grade | N | D | PD | TD_observe | p_binomial | p_jeffreys | feu_jeffreys | rejet_binomial | rejet_vasicek |
|---|---|---|---|---|---|---|---|---|---|
| G01 | 2305 | 3 | 0.05% | 0.13% | 0.1103 | 0.0589 | VERT | False | False |
| G02 | 2581 | 0 | 0.12% | 0.00% | 1.0000 | 0.9872 | VERT | False | False |
| G03 | 2565 | 4 | 0.25% | 0.16% | 0.8823 | 0.8297 | VERT | False | False |
| G04 | 2865 | 13 | 0.45% | 0.45% | 0.5253 | 0.4695 | VERT | False | False |
| G05 | 2903 | 19 | 0.80% | 0.65% | 0.8376 | 0.8087 | VERT | False | False |
| G06 | 3099 | 44 | 1.40% | 1.42% | 0.4832 | 0.4529 | VERT | False | False |
| G07 | 3082 | 73 | 2.50% | 2.37% | 0.6955 | 0.6747 | VERT | False | False |
| G08 | 3201 | 204 | 4.50% | 6.37% | 0.0000 | 0.0000 | ROUGE | True | False |
| G09 | 3513 | 404 | 9.00% | 11.50% | 0.0000 | 0.0000 | ROUGE | True | False |
| G10 | 3837 | 1058 | 20.00% | 27.57% | 0.0000 | 0.0000 | ROUGE | True | False |

Hosmer-Lemeshow : χ² = 198.5, ddl = 10, p-value = 3.36e-37

Lecture : 2025 est simulée comme une année modérément dégradée ET les grades G08 à G10 sont volontairement sous-estimés de 30 %. Le test binomial (défauts indépendants) et le test de Jeffreys rejettent ces grades ; le test ajusté de la corrélation (Vasicek, ρ = 3 %, grand portefeuille) est beaucoup plus tolérant et ne rejette rien. Aucun test seul ne sépare effet de cycle et mauvaise calibration : il faut confronter la PD à la moyenne long terme des taux de défaut sur plusieurs années.

![Calibration](calibration.png)

### Backtesting LGD

| product | n_defauts | LGD_estimee | LGD_realisee | p_value |
|---|---|---|---|---|
| consumer | 631 | 0.4000 | 0.4429 | 0.0000 |
| mortgage | 903 | 0.1500 | 0.2077 | 0.0000 |
| revolving | 288 | 0.5500 | 0.5832 | 0.0001 |

## 4. Stabilité et représentativité

| variable | psi | lecture |
|---|---|---|
| grade | 0.0197 | Stable |
| age | 0.0013 | Stable |
| income | 0.0025 | Stable |
| debt_ratio | 0.2057 | À surveiller |
| seniority_years | 0.0025 | Stable |

Concentration de l'échelle (HHI normalisé, 2025) : 0.0023

## 5. Calculateur RWA (IRB, CRR3)

| product | EAD | RWA | EL | RW_moyen | RWA_CRR2 |
|---|---|---|---|---|---|
| consumer | 178,826,324 | 80,315,029 | 3,351,074 | 44.9% | 85,133,931 |
| mortgage | 2,878,349,196 | 951,837,698 | 20,061,367 | 33.1% | 1,008,947,959 |
| revolving | 16,638,029 | 7,835,950 | 420,619 | 47.1% | 8,286,830 |

Le passage de CRR2 à CRR3 supprime le facteur 1,06 et relève les planchers de PD (0,03 % → 0,05 %, 0,10 % pour les revolving) ; des planchers de LGD s'ajoutent en A-IRB.

### Plancher de sortie (RWA standard illustratifs)

| année | floor_pct | rwa_irb | rwa_floor | rwa_final | binding |
|---|---|---|---|---|---|
| 2025 | 50.0% | 1,039,988,678 | 577,010,242 | 1,039,988,678 | 0 |
| 2027 | 60.0% | 1,039,988,678 | 692,412,290 | 1,039,988,678 | 0 |
| 2030 | 72.5% | 1,039,988,678 | 836,664,851 | 1,039,988,678 | 0 |

### Sensibilité du RW à la PD et à la LGD (PD 2 %, LGD 40 %, autre retail)

| choc | rw_choc_pd | rw_choc_lgd | delta_rw_pd | delta_rw_lgd |
|---|---|---|---|---|
| -20% | 0.4843 | 0.4123 | -0.0605 | -0.2000 |
| -10% | 0.5013 | 0.4639 | -0.0274 | -0.1000 |
| +10% | 0.5272 | 0.5670 | 0.0228 | 0.1000 |
| +20% | 0.5370 | 0.6185 | 0.0418 | 0.2000 |
