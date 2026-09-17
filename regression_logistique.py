

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Chargement uniquement des données d'entraînement.
# Les 101 observations de donnees_test.csv sont réservées au test final.
df = pd.read_csv("donnees_train.csv", sep=None, engine="python")

X = df.drop(columns=["DIFF"])
y = df["DIFF"]

print("Nombre d'observations :", len(df))
print("Variables :", X.columns.tolist())
print("Répartition de DIFF :")
print(y.value_counts())


# Pipeline : la standardisation est apprise séparément
# dans chaque pli afin d'éviter les fuites de données.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=10000,
        random_state=42
    ))
])


# Validation croisée stratifiée à 5 plis.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("\nAUC des 5 plis :", scores)
print("AUC moyenne :", scores.mean())
print("Écart-type :", scores.std())
