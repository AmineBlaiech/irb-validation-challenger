# Méthodes challengers : avantages, limites et cadre d'application

Ce document compare les méthodes implémentées, dans l'esprit d'une fonction de validation qui doit
juger les choix de la modélisation avec des outils indépendants. Il distingue les portefeuilles à
forte volumétrie de défauts (HDP : Retail, PME) et à faible volumétrie (LDP : Grande Clientèle,
souverains, institutions).

## 1. Qualité des données

| Méthode | Ce qu'elle détecte | Avantages | Limites | Cadre d'application |
|---|---|---|---|---|
| Règles déterministes (bornes, formats, cohérence entre champs) | Erreurs certaines : âge 999, EAD < 0, date d'octroi > date d'arrêté, défaut sans date | Exhaustives, explicables, auditables, seuils vert/orange/rouge | Ne trouvent que ce qu'on a prévu ; seuils à justifier | Toujours, en premier contrôle, sur PD, LGD et CCF |
| Tukey / IQR | Valeurs extrêmes univariées | Simple, robuste à quelques valeurs aberrantes | Sur-détecte sur une variable asymétrique (revenu brut : 423 alertes pour 74 erreurs dans la démo) → appliquer sur log(x) | Variables à peu près symétriques ou transformées |
| Score z robuste (médiane, MAD) | Valeurs extrêmes univariées | Le MAD n'est pas contaminé par les valeurs extrêmes, contrairement à l'écart-type | Hypothèse de forme ; MAD nul sur variables très concentrées | Montants, ratios après transformation |
| Percentiles fixes (0,1 % / 99,9 %) | Queues de distribution | Non paramétrique | Signale toujours ~0,2 % de la population même sans erreur | Écrêtage (winsorisation) plutôt que détection |
| Test χ² manquant × défaut | Mécanisme des valeurs manquantes (MCAR ou non) | Dit si supprimer / imputer biaise le modèle | Ne distingue pas MAR et MNAR | Toute variable avec manquants ; décide entre modalité « manquant » et imputation |
| AUC univariée (scan de fuite) | Variables connues après la date d'observation | Rapide, automatique | Une variable très discriminante n'est pas forcément une fuite | Avant toute modélisation ou revue initiale |
| PSI sur les variables d'entrée | Dérive de population, représentativité | Lecture standard (0,10 / 0,25) | Seuils conventionnels ; sensible au découpage | Suivi annuel ; données externes |

## 2. Pouvoir discriminant

| Méthode | Avantages | Limites | Cadre d'application |
|---|---|---|---|
| AUC / Gini | Indépendants de la calibration et du seuil ; standard du superviseur | Peu informatifs avec très peu de défauts ; un Gini élevé ne dit rien du niveau des PD | HDP ; en LDP, compléter par l'analyse experte et les notations externes |
| IC de DeLong | Analytique, rapide | Approximation asymptotique | Échantillons de taille moyenne à grande |
| IC bootstrap | Aucune hypothèse de loi | Coût de calcul ; instable avec très peu de défauts | Challenger de DeLong, petits échantillons |
| Test de dégradation AUC initiale vs courante | Aligné sur le reporting de validation BCE | Ne tient pas compte de la variance de l'AUC initiale | Suivi annuel (backtesting) |

## 3. Calibration

| Méthode | Avantages | Limites | Cadre d'application |
|---|---|---|---|
| Test binomial par grade | Simple, exact | Suppose des défauts indépendants → rejette trop souvent en année de crise | Première lecture, grades bien peuplés |
| Test de Jeffreys | Recommandé par la BCE ; meilleur comportement sur petits effectifs et D = 0 | Même hypothèse d'indépendance | Reporting de validation, tous portefeuilles |
| Binomial ajusté de la corrélation (Vasicek) | Tient compte du cycle | Très tolérant (approximation grand portefeuille) ; ρ à justifier | Challenger du binomial pour séparer cycle et mauvaise calibration |
| Hosmer-Lemeshow | Test global sur tous les grades | Sensible à la taille d'échantillon (rejette tout sur grands portefeuilles) ; ddl à adapter | Vue d'ensemble, à ne pas utiliser seul |
| Comparaison à la moyenne long terme des taux de défaut | Seule manière de juger une PD « à travers le cycle » | Exige un historique long et une définition du défaut stable | Calibration réglementaire (PD long terme) |
| Bornes de Pluto & Tasche | Donnent une PD conservatrice même sans défaut observé | Choix du niveau de confiance | LDP |

## 4. RWA

Le calculateur applique la formule ASRF (CRR art. 153-154), les planchers de PD et de LGD CRR3,
la suppression du facteur 1,06 et le plancher de sortie. Il sert d'outil de challenge : mesurer
l'impact en capital d'une recommandation de validation (ex. : relever la PD d'un grade de 30 %),
comparer CRR2 et CRR3, tester la sensibilité aux paramètres. La sensibilité du RW est quasi
proportionnelle à la LGD et nettement moins que proportionnelle à la PD (+20 % de PD → environ
+4 % de RW pour une exposition retail à PD 2 %) : une erreur de LGD coûte plus cher en capital
qu'une erreur de PD du même ordre.

## Références

- CRR (UE) 575/2013 modifié par le règlement (UE) 2024/1623 (« CRR3 »).
- EBA, *Supervisory handbook on the validation of rating systems under the IRB approach* —
  consultation EBA/CP/2022/08 (juillet 2022), version finale publiée en août 2023.
- EBA/GL/2017/16, orientations sur l'estimation de la PD et de la LGD.
- BCE, *Guide to internal models*, révisions de juillet 2025 et juin 2026.
- BCE, *Instructions for reporting the validation results of internal models – IRB Pillar I
  models for credit risk*, février 2019.
- BCBS 239, principes d'agrégation des données de risque.
- Pluto, K. et Tasche, D. (2005), *Thinking positively*, Risk.
- DeLong, E., DeLong, D. et Clarke-Pearson, D. (1988), Biometrics.
