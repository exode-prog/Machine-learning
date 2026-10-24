"""la version de Python et celle de chaque bibliothèque du projet."""
import sys
from importlib.metadata import version, PackageNotFoundError

BIBLIOTHEQUES = ["numpy", "pandas", "matplotlib", "scikit-learn", "scipy", "joblib"]

print(f"Python       : {sys.version.split()[0]}")
print(f"Interpréteur : {sys.executable}")
print("-" * 40)

for nom in BIBLIOTHEQUES:
    try:
        print(f"{nom:<13}: {version(nom)}")
    except PackageNotFoundError:
        print(f"{nom:<13}: NON INSTALLÉE")