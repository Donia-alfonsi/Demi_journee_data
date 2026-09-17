


import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Chargement uniquement des données d'entraînement.
# Les 101 observations de donnees_test.csv sont réservées au test final.
df = pd.read_csv("donnees_train.csv", sep=None, engine="python")
# Conversion des nombres utilisant une virgule comme séparateur decimal
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False),
            errors="raise"
        )
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
# ---------------------------------------------------------
# Optimisation de la regularisation
# Uniquement sur donnees_train.csv
# ---------------------------------------------------------

from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        "logreg__penalty": ["l2"],
        "logreg__C": [0.001, 0.01, 0.1, 0.3, 0.5, 1, 2, 5, 10, 50, 100],
        "logreg__solver": ["lbfgs"]
    },
    {
        "logreg__penalty": ["l1"],
        "logreg__C": [0.001, 0.01, 0.1, 0.3, 0.5, 1, 2, 5, 10, 50, 100],
        "logreg__solver": ["liblinear"]
    }
]

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    return_train_score=False
)

grid.fit(X, y)

print("\n--- OPTIMISATION ---")
print("Meilleure AUC CV :", grid.best_score_)
print("Meilleurs paramètres :", grid.best_params_)

results = pd.DataFrame(grid.cv_results_)
results = results.sort_values("mean_test_score", ascending=False)

print("\nTop 10 configurations :")
print(
    results[
        ["mean_test_score", "std_test_score",
         "param_logreg__C", "param_logreg__penalty"]
    ].head(10).to_string(index=False)
)
print("\n--- RECHERCHE FINE DE C SANS AGE ---")

X_sans_age = df[["TOF", "R7", "R8", "R17", "R22", "R32"]]

valeurs_C = [
    0.0001, 0.0003, 0.001, 0.003, 0.005,
    0.01, 0.02, 0.03, 0.05, 0.1,
    0.2, 0.3, 0.5, 1, 2, 5, 10
]

for C in valeurs_C:

    modele_c = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=C,
            max_iter=10000,
            random_state=42
        ))
    ])

    scores_c = cross_val_score(
        modele_c,
        X_sans_age,
        y,
        cv=cv,
        scoring="roc_auc"
    )

    print(
        f"C={C:<7} "
        f"AUC={scores_c.mean():.6f} "
        f"std={scores_c.std():.6f}"
    )
