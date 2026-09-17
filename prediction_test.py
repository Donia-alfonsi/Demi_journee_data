import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Chargement TRAIN et TEST
train = pd.read_csv("donnees_train.csv", sep=None, engine="python")
test = pd.read_csv("donnees_test.csv", sep=None, engine="python")
# Suppression des caractères invisibles dans les noms de colonnes
train.columns = train.columns.str.replace("\ufeff", "", regex=False).str.strip()
test.columns = test.columns.str.replace("\ufeff", "", regex=False).str.strip()
# Conversion virgule -> point
for df in [train, test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(
                df[col].str.replace(",", ".", regex=False),
                errors="raise"
            )

# Variables retenues par validation croisee
variables = ["TOF", "R7", "R8", "R17", "R22", "R32"]

X_train = train[variables]
y_train = train["DIFF"]

X_test = test[variables]

# Modele final
modele = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        C=0.01,
        penalty="l2",
        solver="lbfgs",
        max_iter=10000,
        random_state=42
    ))
])

# Entrainement sur la totalite du TRAIN
modele.fit(X_train, y_train)

# Probabilite d'appartenir a DIFF = 1
probas = modele.predict_proba(X_test)[:, 1]

predictions = pd.DataFrame({
    "DIFF": probas
})

predictions.to_csv(
    "predictions_regression_logistique.csv",
    index=False
)

print("Fichier créé : predictions_regression_logistique.csv")
print("Nombre de prédictions :", len(predictions))
print(predictions.head(10))

