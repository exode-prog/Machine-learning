"""
Exercice 1 : régression linéaire (loyer en fonction de la surface).

Exécution : cellule par cellule dans VS Code (Run Cell), ou
             uv run python exo1_regression.py
Les figures sont enregistrées dans le dossier figures/.
"""
# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DOSSIER_FIG = Path("figures")
DOSSIER_FIG.mkdir(exist_ok=True)

# %% Données (graine imposée, à ne pas modifier)
rng = np.random.default_rng(42)
surface = rng.uniform(20, 150, size=60)
loyer = 12 * surface + 150 + rng.normal(0, 120, size=60)


# %% Question 1 : nuage de points
plt.figure(figsize=(7, 5))
plt.scatter(surface, loyer, alpha=0.8, edgecolor="k")
plt.xlabel("Surface (m²)")
plt.ylabel("Loyer")
plt.title("Loyer en fonction de la surface (60 logements)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig1_4_nuage_points.png", dpi=150)
plt.show()

correlation = np.corrcoef(surface, loyer)[0, 1]
print("=== Question 1 : nuage de points ===")
print(f"Coefficient de corrélation : {correlation:.4f}\n")


# %% Question 2 : fonctions vectorisées (sans boucle)
def predire(x, w, b):
    """Prédiction du modèle linéaire : y_chapeau = w * x + b."""
    return w * x + b


def cout_mse(y_vrai, y_predit):
    """Erreur quadratique moyenne : moyenne de (y - y_chapeau)²."""
    return np.mean((y_vrai - y_predit) ** 2)


# %% Question 3 : solution exacte des moindres carrés (équation normale)
X = np.column_stack([np.ones_like(surface), surface])   # colonne de 1 pour b
theta = np.linalg.solve(X.T @ X, X.T @ loyer)            # (XᵀX) θ = Xᵀy
b_exact, w_exact = theta

modele_sk = LinearRegression().fit(surface.reshape(-1, 1), loyer)
w_sk, b_sk = modele_sk.coef_[0], modele_sk.intercept_

print("=== Question 3 : solution exacte vs scikit-learn ===")
print(f"Solution exacte NumPy : w = {w_exact:.6f}  b = {b_exact:.6f}")
print(f"LinearRegression      : w = {w_sk:.6f}  b = {b_sk:.6f}")
print(f"Écart |w| = {abs(w_exact - w_sk):.2e}   Écart |b| = {abs(b_exact - b_sk):.2e}")
print(f"MSE solution exacte   : {cout_mse(loyer, predire(surface, w_exact, b_exact)):.4f}\n")

# Figure : droite ajustée
xs = np.linspace(surface.min(), surface.max(), 100)
plt.figure(figsize=(7, 5))
plt.scatter(surface, loyer, alpha=0.8, edgecolor="k", label="Logements")
plt.plot(xs, predire(xs, w_exact, b_exact), color="red", linewidth=2,
         label=f"Droite ajustée : loyer = {w_exact:.2f} × surface + {b_exact:.1f}")
plt.xlabel("Surface (m²)")
plt.ylabel("Loyer")
plt.title("Régression linéaire par moindres carrés")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig1_5_droite_regression.png", dpi=150)
plt.show()


# %% Question 4 : descente de gradient sur données standardisées
mu_x, sigma_x = surface.mean(), surface.std()
mu_y, sigma_y = loyer.mean(), loyer.std()
x_std = (surface - mu_x) / sigma_x
y_std = (loyer - mu_y) / sigma_y


def gradients(x, y, w, b):
    """Gradients analytiques de la MSE par rapport à w et b."""
    erreur = predire(x, w, b) - y
    dw = 2 * np.mean(erreur * x)
    db = 2 * np.mean(erreur)
    return dw, db


def descente_gradient(x, y, taux, n_iter=200, w0=0.0, b0=0.0):
    """Descente de gradient ; renvoie w, b et l'historique du coût."""
    w, b = w0, b0
    historique = [cout_mse(y, predire(x, w, b))]
    for _ in range(n_iter):
        dw, db = gradients(x, y, w, b)
        w -= taux * dw
        b -= taux * db
        historique.append(cout_mse(y, predire(x, w, b)))
    return w, b, np.array(historique)


TAUX = 0.1
w_s, b_s, cout_hist = descente_gradient(x_std, y_std, TAUX)

# Retour aux unités d'origine
w_orig = w_s * sigma_y / sigma_x
b_orig = mu_y + sigma_y * b_s - w_orig * mu_x

print("=== Question 4 : descente de gradient (taux = 0.1, 200 itérations) ===")
print(f"Paramètres standardisés : w = {w_s:.6f}  b = {b_s:.2e}")
print(f"Unités d'origine        : w = {w_orig:.6f}  b = {b_orig:.6f}")
print(f"Écart avec la solution exacte : |w| = {abs(w_orig - w_exact):.2e}  "
      f"|b| = {abs(b_orig - b_exact):.2e}")
print(f"Coût initial = {cout_hist[0]:.4f}   coût final = {cout_hist[-1]:.6f}\n")

plt.figure(figsize=(7, 5))
plt.plot(cout_hist, linewidth=2)
plt.xlabel("Itération")
plt.ylabel("Coût MSE (données standardisées)")
plt.title(f"Évolution du coût – descente de gradient (taux = {TAUX})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig1_6_cout_iterations.png", dpi=150)
plt.show()


# %% Question 5 : trois taux d'apprentissage
taux_testes = {0.01: "trop petit (lent)", 0.1: "adapté", 1.05: "trop grand (diverge)"}

plt.figure(figsize=(8, 5))
print("=== Question 5 : comparaison des taux d'apprentissage ===")
for taux, etiquette in taux_testes.items():
    w_t, b_t, hist = descente_gradient(x_std, y_std, taux)
    plt.plot(hist, linewidth=2, label=f"taux = {taux} : {etiquette}")
    print(f"taux = {taux:<5} coût final = {hist[-1]:.4e}   w_std = {w_t:.4e}")
plt.yscale("log")
plt.xlabel("Itération")
plt.ylabel("Coût MSE (échelle logarithmique)")
plt.title("Influence du taux d'apprentissage")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(DOSSIER_FIG / "fig1_7_taux_apprentissage.png", dpi=150)
plt.show()
print()


# %% Question 6 : vérification du gradient par différences finies
def gradients_numeriques(x, y, w, b, h=1e-6):
    """Différences finies centrées."""
    dw = (cout_mse(y, predire(x, w + h, b)) - cout_mse(y, predire(x, w - h, b))) / (2 * h)
    db = (cout_mse(y, predire(x, w, b + h)) - cout_mse(y, predire(x, w, b - h))) / (2 * h)
    return dw, db


rng_test = np.random.default_rng(0)
points_test = rng_test.normal(0, 2, size=(20, 2))   # 20 couples (w, b) au hasard
ecarts = []
for w_t, b_t in points_test:
    ana = np.array(gradients(x_std, y_std, w_t, b_t))
    num = np.array(gradients_numeriques(x_std, y_std, w_t, b_t))
    ecarts.append(np.max(np.abs(ana - num)))

print("=== Question 6 : différences finies ===")
print(f"Écart maximal entre gradient analytique et numérique (20 points) : {max(ecarts):.2e}\n")


# %% Métriques
y_pred = predire(surface, w_exact, b_exact)
mae = np.mean(np.abs(loyer - y_pred))
rmse = np.sqrt(np.mean((loyer - y_pred) ** 2))
sce = np.sum((loyer - y_pred) ** 2)
sct = np.sum((loyer - loyer.mean()) ** 2)
r2 = 1 - sce / sct

print("=== Métriques (calcul manuel / scikit-learn) ===")
print(f"MAE  = {mae:.4f}   (sklearn : {mean_absolute_error(loyer, y_pred):.4f})")
print(f"RMSE = {rmse:.4f}   (sklearn : {np.sqrt(mean_squared_error(loyer, y_pred)):.4f})")
print(f"R²   = {r2:.4f}   (sklearn : {r2_score(loyer, y_pred):.4f})")
print(f"Rapport RMSE / MAE = {rmse / mae:.3f}")