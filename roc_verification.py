import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

train = pd.read_csv("donnees_train.csv", sep=None, engine="python")
verif = pd.read_csv("donnees_verification.csv", sep=None, engine="python")

# Nettoyage
for df in [train, verif]:
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(
                df[col].str.replace(",", ".", regex=False),
                errors="raise"
            )

variables = ["TOF", "R7", "R8", "R17", "R22", "R32"]

X_train = train[variables]
y_train = train["DIFF"]

X_verif = verif[variables]
y_verif = verif["DIFF"]

modele = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=10000,
        random_state=42
    ))
])

modele.fit(X_train, y_train)

probas = modele.predict_proba(X_verif)[:, 1]

auc = roc_auc_score(y_verif, probas)

print("AUC VERIFICATION :", auc)

fpr, tpr, _ = roc_curve(y_verif, probas)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2,
         label=f"Régression logistique (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Classification aléatoire")

plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - jeu de vérification")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("figures/roc_verification.png", dpi=200)
plt.close()

print("Figure créée : figures/roc_verification.png")
