import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("donnees_train.csv", sep=None, engine="python")
verification = pd.read_csv(
    "donnees_verification.csv",
    sep=None,
    engine="python"
)

# Nettoyage noms de colonnes
for df in [train, verification]:
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

# Conversion virgule -> point
for df in [train, verification]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(
                df[col].str.replace(",", ".", regex=False),
                errors="raise"
            )

variables = ["TOF", "R7", "R8", "R17", "R22", "R32"]

X_train = train[variables]
y_train = train["DIFF"]
X_verification = verification[variables]

modele = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=10000,
        random_state=42
    ))
])

# Entrainement final
modele.fit(X_train, y_train)

# Prédictions sur TOUT le jeu de vérification
probas = modele.predict_proba(X_verification)[:, 1]

soumission = pd.DataFrame({
    "ID": range(1, len(verification) + 1),
    "DIFF": probas
})

soumission.to_csv(
    "soumission_verification_logistique.csv",
    index=False
)

print("Nombre de prédictions :", len(soumission))
print(soumission.head(10))
