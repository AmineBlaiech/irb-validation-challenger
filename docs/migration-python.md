# Migrer un outil de validation vers une version récente de Python

Ce projet cible Python ≥ 3.12 et a été testé sous Python 3.13 avec NumPy 2.x et pandas 3.x.
Cette note recense ce qui casse le plus souvent quand on migre un outil quantitatif écrit pour
Python 3.7–3.9 et pandas 1.x, et une méthode pour le faire sans changer les résultats.

## Ce qui casse

| Bibliothèque | Changement | Correction |
|---|---|---|
| Python 3.12 | `distutils` supprimé | `setuptools` ou `packaging` |
| NumPy 1.24 | `np.float`, `np.int`, `np.bool` supprimés | `float`, `int`, `bool` |
| NumPy 2.0 | `np.NaN`, `np.Inf`, `np.float_` supprimés ; nouvelles règles de promotion des types (NEP 50) | `np.nan`, `np.inf`, `np.float64` ; revoir les calculs mêlant scalaires et tableaux `float32` |
| pandas 2.0 | `DataFrame.append` et `Series.iteritems` supprimés | `pd.concat`, `.items()` |
| pandas 3.0 | Copy-on-Write par défaut : l'affectation en chaîne `df["a"][masque] = x` ne modifie plus `df` | `df.loc[masque, "a"] = x` |
| pandas 3.0 | Type `str` par défaut pour les colonnes texte | Revoir les tests sur `dtype == object` |

Le cas pandas 3.0 est le plus dangereux pour un outil de validation : le code ne plante pas, il
produit silencieusement un résultat différent.

## Méthode de migration sans régression

1. **Figer une référence** : exécuter l'outil existant sur un jeu de données de test et
   conserver toutes les sorties (indicateurs, tables, p-values).
2. **Environnement neuf** isolé et reproductible (`uv` ou `venv` + fichier de verrouillage).
3. **Faire remonter les alertes** : lancer les tests avec
   `python -W error::DeprecationWarning -W error::FutureWarning`, ce qui transforme chaque
   avertissement de dépréciation en erreur à corriger.
4. **Modernisation automatique** de la syntaxe avec `ruff check --select UP --fix`.
5. **Tests de non-régression** : comparer chaque indicateur à la référence avec une tolérance
   numérique explicite (ex. 1e-10 en absolu sur une AUC) ; tout écart doit être expliqué.
6. **Intégration continue** sur plusieurs versions de Python (voir `.github/workflows/ci.yml`).

## Fonctionnalités récentes utilisées dans ce projet

- `match` / `case` (3.10) pour l'aiguillage par classe d'exposition et dans le CLI.
- `StrEnum` (3.11) pour les classes d'exposition et les dimensions de qualité.
- Instruction `type` (3.12, PEP 695) pour les alias de type.
- `dataclass(slots=True, frozen=True)` pour des objets de configuration immuables.
