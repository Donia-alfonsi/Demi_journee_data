import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Chargement
train = pd.read_csv("farms_train.csv", sep=None, engine="python")
test = pd.read_csv("farms_test.csv", sep=None, engine="python")

# Nettoyage éventuel des noms de colonnes
for df in [train, test]:
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

# Modèle retenu précédemment : AGE retiré
variables = ["TOF", "R7", "R8", "R17", "R22", "R32"]

X_train = train[variables]
y_train = train["DIFF"]
X_test = test[variables]

modele = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=10000,
        random_state=42
    ))
])

# Entraînement sur les 299 observations
modele.fit(X_train, y_train)

# Prédictions finales 0 / 1
predictions = modele.predict(X_test).astype(int)

# CSV demandé
soumission = pd.DataFrame({
    "ID": range(1, len(test) + 1),
    "DIFF": predictions
})

soumission.to_csv("soumission_farms_logistique.csv", index=False)

print("Nombre de prédictions :", len(soumission))
print("\nRépartition :")
print(soumission["DIFF"].value_counts())

print("\nPremières lignes :")
print(soumission.head(10))

print("\nFichier créé : soumission_farms_logistique.csv")
