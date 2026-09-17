# ============================================================
# 1. IMPORT DES DONNÉES
# ============================================================

# Si le fichier est dans ton répertoire de travail :
farms_train <- read.csv("getwd()",
                        header = TRUE,
                        sep = ",")

# Vérifier que les données sont bien importées
head(farms_train)
tail(farms_train)

# ============================================================
# 2. PREMIÈRE EXPLORATION
# ============================================================

# Dimensions du jeu de données
dim(farms_train)

# Nombre de lignes
nrow(farms_train)

# Nombre de colonnes
ncol(farms_train)

# Noms des variables
names(farms_train)

# Structure des données
str(farms_train)

# Résumé statistique de toutes les variables
summary(farms_train)


# ============================================================
# 3. VALEURS MANQUANTES
# ============================================================

# Nombre de valeurs manquantes par variable
colSums(is.na(farms_train))

# Vérifier s'il existe au moins une valeur manquante
any(is.na(farms_train))


# ============================================================
# 4. VARIABLE À PRÉDIRE : DIFF
# ============================================================

# Vérifier les valeurs possibles
unique(farms_train$DIFF)

# Effectifs de chaque classe
table(farms_train$DIFF)

# Proportions de chaque classe
prop.table(table(farms_train$DIFF))

# Graphique de la variable DIFF
barplot(table(farms_train$DIFF),
        main = "Répartition de DIFF",
        xlab = "DIFF",
        ylab = "Effectif")


# ============================================================
# 5. ANALYSE DES VARIABLES EXPLICATIVES
# ============================================================

# Résumé statistique
summary(farms_train)

# Histogrammes des variables numériques
hist(farms_train$TOF,
     main = "Distribution de TOF",
     xlab = "TOF")

hist(farms_train$AGE,
     main = "Distribution de AGE",
     xlab = "AGE")

hist(farms_train$R7,
     main = "Distribution de R7",
     xlab = "R7")

hist(farms_train$R8,
     main = "Distribution de R8",
     xlab = "R8")

hist(farms_train$R17,
     main = "Distribution de R17",
     xlab = "R17")

hist(farms_train$R22,
     main = "Distribution de R22",
     xlab = "R22")

hist(farms_train$R32,
     main = "Distribution de R32",
     xlab = "R32")


# ============================================================
# 6. RELATION ENTRE LES VARIABLES ET DIFF
# ============================================================

# Comparer les distributions selon DIFF

boxplot(TOF ~ DIFF,
        data = farms_train,
        main = "TOF selon DIFF",
        xlab = "DIFF",
        ylab = "TOF")

boxplot(AGE ~ DIFF,
        data = farms_train,
        main = "AGE selon DIFF",
        xlab = "DIFF",
        ylab = "AGE")

boxplot(R7 ~ DIFF,
        data = farms_train,
        main = "R7 selon DIFF",
        xlab = "DIFF",
        ylab = "R7")

boxplot(R8 ~ DIFF,
        data = farms_train,
        main = "R8 selon DIFF",
        xlab = "DIFF",
        ylab = "R8")

boxplot(R17 ~ DIFF,
        data = farms_train,
        main = "R17 selon DIFF",
        xlab = "DIFF",
        ylab = "R17")

boxplot(R22 ~ DIFF,
        data = farms_train,
        main = "R22 selon DIFF",
        xlab = "DIFF",
        ylab = "R22")

boxplot(R32 ~ DIFF,
        data = farms_train,
        main = "R32 selon DIFF",
        xlab = "DIFF",
        ylab = "R32")


# ============================================================
# 7. MATRICE DE CORRÉLATIONS
# ============================================================

# Corrélations entre variables quantitatives
cor(farms_train[, c("TOF", "AGE", "R7", "R8",
                    "R17", "R22", "R32")],
    use = "complete.obs")


# ============================================================
# 8. VISUALISATION DES RELATIONS ENTRE VARIABLES
# ============================================================

pairs(farms_train[, c("TOF", "AGE", "R7", "R8",
                      "R17", "R22", "R32")],
      main = "Relations entre les variables")


# ============================================================
# 9. PREMIÈRE ANALYSE PAR GROUPE DE DIFF
# ============================================================

# Moyennes des variables selon DIFF
aggregate(cbind(TOF, AGE, R7, R8, R17, R22, R32) ~ DIFF,
          data = farms_train,
          FUN = mean)

# Médianes selon DIFF
aggregate(cbind(TOF, AGE, R7, R8, R17, R22, R32) ~ DIFF,
          data = farms_train,
          FUN = median)


# ============================================================
# 10. À CE STADE : PRÉPARATION POUR LA CLASSIFICATION
# ============================================================

# Vérifier le type de DIFF
class(farms_train$DIFF)

# Si DIFF est codée 0/1, c'est adapté à une classification
table(farms_train$DIFF)

# ============================================================
# FIN DE L'ANALYSE EXPLORATOIRE
# ============================================================
