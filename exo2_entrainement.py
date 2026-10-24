"""
Exercice 2.1 : entraîner, évaluer et sauvegarder un modèle de progression du diabète.

Exécution : cellule par cellule dans VS Code (Run Cell), ou
            uv run python exo2_entrainement.py
Produit : modeles/diabete.joblib, modeles/diabete_infos.json et deux figures dans figures/.
"""
# %% Imports
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DOSSIER_FIG = Path("figures")
DOSSIER_MODELES = Path("modeles")
DOSSIER_FIG.mkdir(exist_ok=True)
DOSSIER_MODELES.mkdir(exist_ok=True)

GRAINE = 42


def metriques(y_vrai, y_predit):
    """Renvoie MAE, RMSE et R² dans un dictionnaire."""
    return {
        "MAE": mean_absolute_error(y_vrai, y_predit),
        "RMSE": np.sqrt(mean_squared_error(y_vrai, y_predit)),
        "R2": r2_score(y_vrai, y_predit),
    }


# %% Question 1 : chargement et description des données
# scaled=False : on garde les mesures dans leurs unités d'origine (âge en années,
# IMC, tension...), comme elles arriveraient d'un vrai dossier patient.
donnees = load_diabetes(as_frame=True, scaled=False)
X = donnees.data
y = donnees.target

print("=== Question 1 : description des données ===")
print(f"Forme de X : {X.shape}")
print(f"Forme de y : {y.shape}")
print(f"Variables  : {list(X.columns)}")
print("\nAperçu des 5 premiers patients :")
print(X.head())
print("\nStatistiques de la cible (progression à un an) :")
print(y.describe().round(1))


# %% Question 2 : découpage entraînement / test AVANT toute transformation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=GRAINE
)
print("=== Question 2 : découpage ===")
print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test         : {X_test.shape[0]} patients")


# %% Question 3 : pipeline StandardScaler + LinearRegression
pipeline_lin = Pipeline([
    ("standardisation", StandardScaler()),
    ("regression", LinearRegression()),
])
print("=== Question 3 : pipeline ===")
print(pipeline_lin)


# %% Question 4 : validation croisée à 5 plis, puis évaluation unique sur le test
plis = KFold(n_splits=5, shuffle=True, random_state=GRAINE)
SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}


def validation_croisee(modele):
    """Validation croisée à 5 plis sur l'entraînement : moyenne et écart-type."""
    res = cross_validate(modele, X_train, y_train, cv=plis, scoring=SCORING)
    resultats = {}
    for nom in SCORING:
        valeurs = res[f"test_{nom}"]
        if nom in ("MAE", "RMSE"):
            valeurs = -valeurs            # scikit-learn renvoie des scores négatifs
        resultats[nom] = (valeurs.mean(), valeurs.std())
    return resultats


cv_lin = validation_croisee(pipeline_lin)
pipeline_lin.fit(X_train, y_train)
test_lin = metriques(y_test, pipeline_lin.predict(X_test))

print("=== Question 4 : LinearRegression ===")
print(f"{'Métrique':<6} {'Validation croisée (moy ± é-t)':>32} {'Test':>10}")
for nom in SCORING:
    moy, et = cv_lin[nom]
    print(f"{nom:<6} {moy:>22.3f} ± {et:<7.3f} {test_lin[nom]:>10.3f}")


# %% Question 5 : comparaison avec Ridge, alpha choisi par validation croisée
pipeline_ridge = Pipeline([
    ("standardisation", StandardScaler()),
    ("regression", Ridge()),
])
grille_alpha = np.logspace(-3, 3, 61)          # de 0,001 à 1000
recherche = GridSearchCV(
    pipeline_ridge,
    {"regression__alpha": grille_alpha},
    cv=plis,
    scoring="neg_root_mean_squared_error",
)
recherche.fit(X_train, y_train)
alpha_retenu = recherche.best_params_["regression__alpha"]

meilleur_ridge = recherche.best_estimator_
cv_ridge = validation_croisee(meilleur_ridge)
test_ridge = metriques(y_test, meilleur_ridge.predict(X_test))

print("=== Question 5 : Ridge ===")
print(f"Alpha retenu par validation croisée : {alpha_retenu:.4f}")
print(f"{'Métrique':<6} {'Linéaire (CV)':>14} {'Ridge (CV)':>12} {'Linéaire (test)':>16} {'Ridge (test)':>13}")
for nom in SCORING:
    print(f"{nom:<6} {cv_lin[nom][0]:>14.3f} {cv_ridge[nom][0]:>12.3f} "
          f"{test_lin[nom]:>16.3f} {test_ridge[nom]:>13.3f}")
gain = cv_lin["RMSE"][0] - cv_ridge["RMSE"][0]
print(f"\nGain de RMSE en validation croisée : {gain:.3f} "
      f"({100 * gain / cv_lin['RMSE'][0]:.2f} %)")

# Figure : RMSE de validation croisée en fonction de alpha
rmse_cv = -recherche.cv_results_["mean_test_score"]
plt.figure(figsize=(7, 5))
plt.semilogx(grille_alpha, rmse_cv, linewidth=2)
plt.axvline(alpha_retenu, color="red", linestyle="--",
            label=f"α retenu = {alpha_retenu:.2f}")
plt.axhline(cv_lin["RMSE"][0], color="grey", linestyle=":",
            label="LinearRegression (sans pénalité)")
plt.xlabel("α (échelle logarithmique)")
plt.ylabel("RMSE moyenne en validation croisée")
plt.title("Choix de α pour le modèle Ridge")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig2_2_ridge_alpha.png", dpi=150)
plt.show()


# %% Question 6 : valeur prédite contre valeur réelle sur le test
y_pred_test = pipeline_lin.predict(X_test)
bornes = [min(y_test.min(), y_pred_test.min()) - 10, max(y_test.max(), y_pred_test.max()) + 10]

plt.figure(figsize=(6.5, 6))
plt.scatter(y_test, y_pred_test, alpha=0.8, edgecolor="k", label="Patients du test")
plt.plot(bornes, bornes, color="red", linewidth=2, label="Prédiction parfaite (y = ŷ)")
plt.xlim(bornes)
plt.ylim(bornes)
plt.xlabel("Progression réelle")
plt.ylabel("Progression prédite")
plt.title(f"Prédit contre réel (test) – R² = {test_lin['R2']:.3f}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig2_3_predit_vs_reel.png", dpi=150)
plt.show()

residus = y_test - y_pred_test
print("=== Question 6 : dispersion ===")
print(f"Plus grande sous-estimation : {residus.max():.1f}")
print(f"Plus grande surestimation   : {residus.min():.1f}")


# %% Question 7 : sauvegarde du pipeline complet et des métadonnées
chemin_modele = DOSSIER_MODELES / "diabete.joblib"
joblib.dump(pipeline_lin, chemin_modele)

empreinte = hashlib.sha256(chemin_modele.read_bytes()).hexdigest()

infos = {
    "date_entrainement": datetime.now().isoformat(timespec="seconds"),
    "modele": "Pipeline StandardScaler + LinearRegression",
    "versions": {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
    },
    "variables_attendues": list(X.columns),
    "unites": "mesures non standardisées (load_diabetes(scaled=False))",
    "n_entrainement": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "metriques_test": {nom: round(float(val), 4) for nom, val in test_lin.items()},
    "sha256_modele": empreinte,
}
chemin_infos = DOSSIER_MODELES / "diabete_infos.json"
chemin_infos.write_text(json.dumps(infos, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== Question 7 : sauvegarde ===")
print(f"Modèle enregistré       : {chemin_modele}")
print(f"Informations enregistrées : {chemin_infos}")
print(json.dumps(infos, indent=2, ensure_ascii=False))
# %%


