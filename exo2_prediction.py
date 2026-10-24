"""
Exercice 2.2 : mise en production du modèle de progression du diabète.

Ce script ne réentraîne rien et ne charge pas le jeu de données complet :
il recharge le pipeline sauvegardé et prédit pour de nouveaux patients.

Exécution : cellule par cellule dans VS Code (Run Cell), ou
            uv run python exo2_prediction.py
Prérequis : avoir exécuté exo2_entrainement.py (dossier modeles/)
            et disposer du fichier nouveaux_patients.csv.
"""
# %% Imports et chemins
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

CHEMIN_MODELE = Path("modeles") / "diabete.joblib"
CHEMIN_INFOS = Path("modeles") / "diabete_infos.json"
CHEMIN_CSV = Path("nouveaux_patients.csv")


# %% Question 1 : rechargement du pipeline (après contrôle d'intégrité)
infos = json.loads(CHEMIN_INFOS.read_text(encoding="utf-8"))

# Parade contre un fichier modifié ou piégé : on vérifie l'empreinte SHA-256
# enregistrée à l'entraînement AVANT d'appeler joblib.load.
empreinte = hashlib.sha256(CHEMIN_MODELE.read_bytes()).hexdigest()
if empreinte != infos["sha256_modele"]:
    raise RuntimeError(
        "L'empreinte du fichier modèle ne correspond pas à celle enregistrée : "
        "fichier modifié ou non fiable, chargement refusé."
    )

modele = joblib.load(CHEMIN_MODELE)
print("=== Question 1 : rechargement ===")
print("Empreinte SHA-256 vérifiée : OK")
print(f"Modèle rechargé : {modele}")
print(f"Entraîné le     : {infos['date_entrainement']}")


# %% Question 2 : contrôle de la version de scikit-learn
version_enregistree = infos["versions"]["scikit-learn"]
version_poste = sklearn.__version__

print("=== Question 2 : contrôle de version ===")
print(f"scikit-learn à l'entraînement : {version_enregistree}")
print(f"scikit-learn sur ce poste     : {version_poste}")
if version_poste != version_enregistree:
    warnings.warn(
        f"ATTENTION : le modèle a été entraîné avec scikit-learn {version_enregistree} "
        f"mais ce poste utilise la version {version_poste}. Les prédictions peuvent "
        f"être fausses ou le chargement instable. Réinstallez la version d'origine "
        f"(uv sync) ou réentraînez le modèle.",
        UserWarning,
    )
else:
    print("Versions identiques : OK")

VARIABLES = infos["variables_attendues"]
RMSE_TEST = infos["metriques_test"]["RMSE"]


def afficher_prediction(prediction, nom="Patient"):
    """Question 5 : prédiction accompagnée d'une marge fondée sur la RMSE de test."""
    print(f"{nom} : progression estimée à {prediction:.0f}, à ± {RMSE_TEST:.0f} près "
          f"(intervalle indicatif : {prediction - RMSE_TEST:.0f} à {prediction + RMSE_TEST:.0f})")


# %% Question 3 : prédiction pour un nouveau patient (dictionnaire)
nouveau_patient = {
    "age": 58, "sex": 2, "bmi": 30.2, "bp": 102,
    "s1": 205, "s2": 125.0, "s3": 42, "s4": 5.0, "s5": 5.0, "s6": 95,
}

# On construit le tableau en respectant l'ordre des variables enregistré
patient_df = pd.DataFrame([nouveau_patient])[VARIABLES]

print("=== Question 3 : un patient ===")
print(patient_df.to_string(index=False))
prediction = modele.predict(patient_df)[0]
afficher_prediction(prediction, "Nouveau patient")


# %% Question 4 : prédiction pour trois patients lus dans un fichier CSV
patients = pd.read_csv(CHEMIN_CSV)

manquantes = set(VARIABLES) - set(patients.columns)
if manquantes:
    raise ValueError(f"Colonnes manquantes dans le CSV : {sorted(manquantes)}")

patients = patients[VARIABLES]                  # remise dans l'ordre attendu
predictions = modele.predict(patients)

print("=== Question 4 : trois patients (CSV) ===")
print(patients.to_string())
print()

# %% Question 5 : affichage avec la marge d'incertitude
print("=== Question 5 : prédictions avec marge d'incertitude ===")
afficher_prediction(prediction, "Patient du dictionnaire")
for i, pred in enumerate(predictions, start=1):
    afficher_prediction(pred, f"Patient {i} du CSV     ")