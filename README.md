# Examen bloc A : machine learning

Projet réalisé dans le cadre de l'examen de machine learning 

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `pyproject.toml` | Déclaration du projet et des dépendances |
| `uv.lock` | Versions exactes de toutes les bibliothèques (environnement verrouillé) |
| `.python-version` | Version de Python du projet (3.12) |
| `environnement.py` | Affiche la version de Python et des bibliothèques installées |
| `regression.py` | Exercice 1 : régression linéaire (loyer en fonction de la surface) |
| `exo2_entrainement.py` | Exercice 2 : entraînement, évaluation et sauvegarde du modèle diabète |
| `exo2_prediction.py` | Exercice 2 : rechargement du modèle et prédiction pour de nouveaux patients |
| `nouveaux_patients.csv` | Mesures de trois nouveaux patients pour `exo2_prediction.py` |
| `exo3_fraude.py` | Exercice 3 : détection de fraude sur données déséquilibrées |
| `figures/` | Figures produites par les scripts |
| `modeles/` | Modèle sauvegardé (`diabete.joblib`) et ses métadonnées (`diabete_infos.json`) |

## Reproduire le projet

### 1. Prérequis
Installer [uv](https://docs.astral.sh/uv/). uv installe lui-même Python 3.12 si nécessaire.

### 2. Récupérer le projet et recréer l'environnement
```bash
git clone https://github.com/exode-prog/Machine-learning.git
cd Machine-learning
uv sync --locked
```
`uv sync --locked` crée l'environnement virtuel `.venv` et installe **exactement** les versions
inscrites dans `uv.lock`.

### 3. Vérifier l'environnement
```bash
uv run python environnement.py
```

### 4. Lancer les exercices, dans cet ordre
```bash
uv run python regression.py
uv run python exo2_entrainement.py
uv run python exo2_prediction.py
uv run python exo3_fraude.py
```
`exo2_prediction.py` doit être lancé **après** `exo2_entrainement.py`, qui crée le modèle dans `modeles/`.

Les scripts sont aussi découpés en cellules (`# %%`) : on peut les exécuter question par question
dans VS Code avec l'extension Jupyter (bouton **Run Cell**).

## Reproductibilité
- **Graines aléatoires fixées** (`np.random.default_rng(42)`, `random_state=42`) : mêmes données, mêmes découpages, mêmes résultats.
- **Environnement verrouillé** avec `uv.lock` : mêmes versions de bibliothèques sur tous les postes.
- **Métadonnées du modèle** dans `modeles/diabete_infos.json` : date, versions, ordre des variables, métriques et empreinte SHA-256.

## Sécurité
Un fichier `.joblib` peut exécuter du code au chargement. `exo2_prediction.py` vérifie l'empreinte
SHA-256 du modèle avant de le charger. Ne chargez jamais un fichier `.joblib` dont vous ne connaissez pas la provenance.