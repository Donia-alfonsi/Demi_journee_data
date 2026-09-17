import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Chargement des données
# =========================

df = pd.read_csv("donnees_train.csv", sep=None, engine="python")

# Conversion des nombres écrits avec une virgule
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False),
            errors="raise"
        )

os.makedirs("figures", exist_ok=True)

print("Dimensions :", df.shape)
print("\nRépartition de DIFF :")
print(df["DIFF"].value_counts())

# =========================
# 1. Répartition des classes
# =========================

counts = df["DIFF"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(["DIFF = 0", "DIFF = 1"], counts.values)
plt.title("Répartition de la variable cible DIFF")
plt.ylabel("Nombre d'observations")
plt.tight_layout()
plt.savefig("figures/repartition_DIFF.png", dpi=200)
plt.close()

# =========================
# 2. Matrice de corrélation
# =========================

corr = df.corr(method="spearman")

plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Corrélation de Spearman")
plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45,
    ha="right"
)
plt.yticks(
    range(len(corr.columns)),
    corr.columns
)
plt.title("Matrice de corrélation de Spearman")
plt.tight_layout()
plt.savefig("figures/correlation_spearman.png", dpi=200)
plt.close()

# =========================
# 3. Distribution de R32
# selon la classe
# =========================

r32_0 = df.loc[df["DIFF"] == 0, "R32"]
r32_1 = df.loc[df["DIFF"] == 1, "R32"]

plt.figure(figsize=(7, 4))
plt.hist(r32_0, bins=20, alpha=0.6, label="DIFF = 0")
plt.hist(r32_1, bins=20, alpha=0.6, label="DIFF = 1")
plt.xlabel("R32")
plt.ylabel("Nombre d'observations")
plt.title("Distribution de R32 selon DIFF")
plt.legend()
plt.tight_layout()
plt.savefig("figures/distribution_R32.png", dpi=200)
plt.close()

# =========================
# 4. R17 et R32 selon DIFF
# =========================

plt.figure(figsize=(7, 5))

classe0 = df[df["DIFF"] == 0]
classe1 = df[df["DIFF"] == 1]

plt.scatter(
    classe0["R17"],
    classe0["R32"],
    alpha=0.6,
    label="DIFF = 0"
)

plt.scatter(
    classe1["R17"],
    classe1["R32"],
    alpha=0.6,
    label="DIFF = 1"
)

plt.xlabel("R17")
plt.ylabel("R32")
plt.title("Relation entre R17 et R32 selon DIFF")
plt.legend()
plt.tight_layout()
plt.savefig("figures/R17_R32_DIFF.png", dpi=200)
plt.close()

print("\nVisualisations créées dans le dossier figures/")
# =========================
# 5. Influence du parametre C
# =========================

valeurs_C = [
    0.0001, 0.0003, 0.001, 0.003, 0.005,
    0.01, 0.02, 0.03, 0.05, 0.1,
    0.2, 0.3, 0.5, 1, 2, 5, 10
]

auc_C = [
    0.902697, 0.903142, 0.903594, 0.903594, 0.903602,
    0.904506, 0.903402, 0.903632, 0.902751, 0.903663,
    0.902352, 0.901908, 0.901456, 0.900567, 0.900130,
    0.899019, 0.899241
]

plt.figure(figsize=(8, 5))

plt.plot(
    valeurs_C,
    auc_C,
    marker="o"
)

plt.xscale("log")

# Mise en évidence du meilleur résultat
plt.scatter(
    [0.01],
    [0.904506],
    s=100,
    label="Meilleur : C = 0.01"
)

plt.axvline(
    x=0.01,
    linestyle="--",
    alpha=0.6
)

plt.xlabel("Paramètre C (échelle logarithmique)")
plt.ylabel("AUC moyenne en validation croisée")
plt.title("Influence de C sur la régression logistique")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/auc_selon_C.png",
    dpi=200
)

plt.close()
# =========================
# 6. Courbe ROC en validation croisee
# =========================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

# Modele retenu : sans AGE, C = 0.01
X_roc = df[["TOF", "R7", "R8", "R17", "R22", "R32"]]
y_roc = df["DIFF"]

cv_roc = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

modele_roc = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        C=0.01,
        penalty="l2",
        solver="lbfgs",
        max_iter=10000,
        random_state=42
    ))
])

# Probabilites obtenues hors du pli d'entrainement
probas = cross_val_predict(
    modele_roc,
    X_roc,
    y_roc,
    cv=cv_roc,
    method="predict_proba"
)[:, 1]

auc_roc = roc_auc_score(y_roc, probas)
fpr, tpr, _ = roc_curve(y_roc, probas)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Régression logistique (AUC = {auc_roc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Classification aléatoire"
)

plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - validation croisée")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures/roc_regression_logistique.png",
    dpi=200
)

plt.close()

print(f"AUC ROC out-of-fold : {auc_roc:.6f}")
