"""
Exercice 3 : classification déséquilibrée (détection de transactions frauduleuses).

Exécution : cellule par cellule dans VS Code (Run Cell), ou
            uv run python exo3_fraude.py
Les figures sont enregistrées dans le dossier figures/.
"""
# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DOSSIER_FIG = Path("figures")
DOSSIER_FIG.mkdir(exist_ok=True)

GRAINE = 42
COUT_FRAUDE_MANQUEE = 200     # faux négatif
COUT_FAUSSE_ALERTE = 10       # faux positif


def cout_total(y_vrai, y_predit):
    """Coût métier : 200 par fraude manquée + 10 par fausse alerte."""
    vn, fp, fn, vp = confusion_matrix(y_vrai, y_predit, labels=[0, 1]).ravel()
    return COUT_FRAUDE_MANQUEE * fn + COUT_FAUSSE_ALERTE * fp


# %% Données (code de l'énoncé, graine inchangée)
rng = np.random.default_rng(42)
n = 20_000
tx = pd.DataFrame({
    "montant": rng.lognormal(3.5, 1.0, n).round(2),
    "heure": rng.integers(0, 24, n),
    "distance_km": rng.exponential(10, n).round(1),
    "nb_transactions_24h": rng.poisson(2, n),
    "etranger": (rng.random(n) < 0.05).astype(int),
})
nuit = ((tx["heure"] < 6) | (tx["heure"] >= 23)).astype(int)
logit = (-11.5 + np.log(tx["montant"]) + 2.0 * nuit + 0.08 * tx["distance_km"]
         + 0.5 * tx["nb_transactions_24h"] + 3.0 * tx["etranger"])
tx["fraude"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)


# %% Question 1 : taux de fraude, variables log_montant et nuit
taux_fraude = tx["fraude"].mean()
print("=== Question 1 ===")
print(f"Nombre de transactions : {len(tx)}")
print(f"Nombre de fraudes      : {tx['fraude'].sum()}")
print(f"Taux de fraude         : {100 * taux_fraude:.2f} %")

tx["log_montant"] = np.log(tx["montant"])
tx["nuit"] = nuit

print("\nDistribution des montants :")
print(f"  moyenne = {tx['montant'].mean():.1f}   médiane = {tx['montant'].median():.1f}   "
      f"max = {tx['montant'].max():.1f}")
print(f"  asymétrie (skewness) : montant = {tx['montant'].skew():.2f}   "
      f"log_montant = {tx['log_montant'].skew():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(tx["montant"], bins=80, edgecolor="k", alpha=0.8)
axes[0].set_title("Montant brut")
axes[0].set_xlabel("Montant")
axes[0].set_ylabel("Nombre de transactions")
axes[1].hist(tx["log_montant"], bins=60, edgecolor="k", alpha=0.8, color="tab:orange")
axes[1].set_title("Logarithme du montant")
axes[1].set_xlabel("log(montant)")
for ax in axes:
    ax.grid(alpha=0.3)
plt.suptitle("Distribution des montants avant et après passage au logarithme")
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig3_1_distribution_montants.png", dpi=150)
plt.show()

VARIABLES = ["log_montant", "nuit", "distance_km", "nb_transactions_24h", "etranger"]
X = tx[VARIABLES]
y = tx["fraude"]


# %% Question 2 : découpage stratifié 75 / 25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=GRAINE
)
print("=== Question 2 : découpage stratifié ===")
print(f"Entraînement : {len(y_train)} transactions, {y_train.sum()} fraudes "
      f"({100 * y_train.mean():.2f} %)")
print(f"Test         : {len(y_test)} transactions, {y_test.sum()} fraudes "
      f"({100 * y_test.mean():.2f} %)")


# %% Question 3 : modèle naïf « toujours légitime »
naif = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
y_naif = naif.predict(X_test)
print("=== Question 3 : modèle naïf ===")
print(f"Exactitude sur le test : {100 * accuracy_score(y_test, y_naif):.2f} %")
print(f"Fraudes détectées      : {((y_naif == 1) & (y_test == 1)).sum()} sur {y_test.sum()}")
print(f"Rappel                 : {recall_score(y_test, y_naif):.2f}")


# %% Question 4 : régression logistique en pipeline, probabilités honnêtes
plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)
pipeline_log = Pipeline([
    ("standardisation", StandardScaler()),
    ("logistique", LogisticRegression(max_iter=1000)),
])

# Chaque probabilité est prédite par un modèle qui n'a pas vu cette transaction
proba_cv_log = cross_val_predict(
    pipeline_log, X_train, y_train, cv=plis, method="predict_proba"
)[:, 1]
print("=== Question 4 : probabilités par validation croisée ===")
print(f"Probabilité moyenne prédite : {proba_cv_log.mean():.4f} "
      f"(taux réel : {y_train.mean():.4f})")
print(f"ROC-AUC (validation croisée) : {roc_auc_score(y_train, proba_cv_log):.3f}")
print(f"PR-AUC  (validation croisée) : {average_precision_score(y_train, proba_cv_log):.3f}")


# %% Question 5 : matrice de confusion au seuil 0,5
y_cv_05 = (proba_cv_log >= 0.5).astype(int)
vn, fp, fn, vp = confusion_matrix(y_train, y_cv_05).ravel()
print("=== Question 5 : matrice de confusion au seuil 0,5 (validation croisée) ===")
print(f"Vrais négatifs  (VN) : {vn:>6}  transactions légitimes laissées passer")
print(f"Faux positifs   (FP) : {fp:>6}  fausses alertes (client bloqué à tort)")
print(f"Faux négatifs   (FN) : {fn:>6}  fraudes manquées")
print(f"Vrais positifs  (VP) : {vp:>6}  fraudes détectées")
print(f"Coût au seuil 0,5    : {cout_total(y_train, y_cv_05):_}".replace("_", " "))

disp = ConfusionMatrixDisplay(
    confusion_matrix(y_train, y_cv_05), display_labels=["Légitime", "Fraude"]
)
disp.plot(cmap="Blues", values_format="d")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion au seuil 0,5")
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig3_2_matrice_confusion_05.png", dpi=150)
plt.show()


# %% Question 6 : coût total en fonction du seuil
seuils = np.round(np.arange(0.01, 1.00, 0.01), 2)


def courbe_cout(y_vrai, proba):
    return np.array([cout_total(y_vrai, (proba >= s).astype(int)) for s in seuils])


couts_log = courbe_cout(y_train, proba_cv_log)
seuil_opt_log = seuils[np.argmin(couts_log)]
cout_05 = couts_log[seuils == 0.5][0]
cout_opt = couts_log.min()

print("=== Question 6 : choix du seuil (validation croisée) ===")
print(f"Seuil optimal       : {seuil_opt_log:.2f}  -> coût {cout_opt:_}".replace("_", " "))
print(f"Seuil par défaut 0,5 :      -> coût {cout_05:_}".replace("_", " "))
print(f"Seuil théorique 10 / (10 + 200) = {COUT_FAUSSE_ALERTE / (COUT_FAUSSE_ALERTE + COUT_FRAUDE_MANQUEE):.3f}")

plt.figure(figsize=(8, 5))
plt.plot(seuils, couts_log, linewidth=2)
plt.axvline(seuil_opt_log, color="green", linestyle="--",
            label=f"Seuil optimal = {seuil_opt_log:.2f}")
plt.axvline(0.5, color="red", linestyle=":", label="Seuil par défaut = 0,5")
plt.xlabel("Seuil de décision")
plt.ylabel("Coût total (validation croisée, entraînement)")
plt.title("Coût total en fonction du seuil – régression logistique")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig3_3_cout_seuil.png", dpi=150)
plt.show()


# %% Question 7 : application unique au jeu de test
pipeline_log.fit(X_train, y_train)
proba_test_log = pipeline_log.predict_proba(X_test)[:, 1]
y_test_opt = (proba_test_log >= seuil_opt_log).astype(int)

cout_sans_detection = COUT_FRAUDE_MANQUEE * y_test.sum()   # toutes les fraudes passent
cout_test_opt = cout_total(y_test, y_test_opt)
cout_test_05 = cout_total(y_test, (proba_test_log >= 0.5).astype(int))

vn, fp, fn, vp = confusion_matrix(y_test, y_test_opt).ravel()
print("=== Question 7 : test, seuil optimal appliqué une seule fois ===")
print(f"Seuil appliqué : {seuil_opt_log:.2f}")
print(f"VN = {vn}  FP = {fp}  FN = {fn}  VP = {vp}")
print(f"Coût sans détection       : {cout_sans_detection:_}".replace("_", " "))
print(f"Coût au seuil 0,5         : {cout_test_05:_}".replace("_", " "))
print(f"Coût au seuil optimal     : {cout_test_opt:_}".replace("_", " "))
print(f"Coût évité                : {cout_sans_detection - cout_test_opt:_} "
      f"({100 * (1 - cout_test_opt / cout_sans_detection):.1f} %)".replace("_", " "))


# %% Question 8 : arbre de décision réglé et forêt aléatoire
# Les arbres comparent chaque variable à un seuil : pas besoin de standardiser.
grille_arbre = GridSearchCV(
    DecisionTreeClassifier(random_state=GRAINE),
    {"max_depth": [2, 3, 4, 5, 6, 8, 10], "min_samples_leaf": [10, 50, 100, 200]},
    cv=plis, scoring="average_precision",
)
grille_arbre.fit(X_train, y_train)
arbre = grille_arbre.best_estimator_
print("=== Question 8 : modèles à base d'arbres ===")
print(f"Arbre de décision, meilleurs réglages : {grille_arbre.best_params_}")

foret = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=20, random_state=GRAINE, n_jobs=-1
)

modeles = {"Régression logistique": pipeline_log, "Arbre de décision": arbre,
           "Forêt aléatoire": foret}
resultats = {}
for nom, modele in modeles.items():
    # seuil choisi sur des probabilités de validation croisée (entraînement)
    proba_cv = cross_val_predict(modele, X_train, y_train, cv=plis,
                                 method="predict_proba")[:, 1]
    seuil = seuils[np.argmin(courbe_cout(y_train, proba_cv))]
    # puis une seule évaluation sur le test
    modele.fit(X_train, y_train)
    proba_test = modele.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= seuil).astype(int)
    resultats[nom] = {
        "proba": proba_test,
        "Seuil": seuil,
        "Exactitude": accuracy_score(y_test, y_pred),
        "Précision": precision_score(y_test, y_pred, zero_division=0),
        "Rappel": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, proba_test),
        "PR-AUC": average_precision_score(y_test, proba_test),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Coût": cout_total(y_test, y_pred),
    }

tableau = pd.DataFrame({nom: {k: v for k, v in r.items() if k != "proba"}
                        for nom, r in resultats.items()}).T
print("\nComparaison sur le test (chaque modèle à son seuil optimal) :")
print(tableau.round(3).to_string())


# %% Question 9 : courbes ROC et précision-rappel côte à côte
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
for nom, r in resultats.items():
    RocCurveDisplay.from_predictions(y_test, r["proba"], ax=ax_roc,
                                     name=f"{nom}")
    PrecisionRecallDisplay.from_predictions(y_test, r["proba"], ax=ax_pr,
                                            name=f"{nom}")
ax_roc.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0,5)")
ax_roc.set_title("Courbe ROC")
ax_roc.set_xlabel("Taux de faux positifs")
ax_roc.set_ylabel("Rappel (taux de vrais positifs)")
ax_pr.axhline(y_test.mean(), color="k", linestyle="--",
              label=f"Hasard (= taux de fraude {100 * y_test.mean():.1f} %)")
ax_pr.set_title("Courbe précision-rappel")
ax_pr.set_xlabel("Rappel")
ax_pr.set_ylabel("Précision")
for ax in (ax_roc, ax_pr):
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig3_4_roc_pr.png", dpi=150)
plt.show()