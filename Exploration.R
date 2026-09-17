# ============================================================
# 1. IMPORT DES DONNÉES
# ============================================================

train <- read.csv2("../farms_train.csv")

head(train)
tail(train)

colonnes_num <- c("TOF", "AGE", "R7", "R8", "R17", "R22", "R32")

for (col in colonnes_num) {
  if (!is.numeric(train[[col]])) {
    train[[col]] <- as.numeric(gsub(",", ".", as.character(train[[col]])))
  }
}

train$DIFF <- as.factor(train$DIFF)

# Dimensions du jeu de données
dim(train)

# Nombre de lignes
nrow(train)

# Nombre de colonnes
ncol(train)

# Noms des variables
names(train)

# Structure des données
str(train)

# Résumé statistique de toutes les variables
summary(train)



# Nombre de valeurs manquantes par variable
colSums(is.na(train))

# ============================================================
# 4. VARIABLE À PRÉDIRE : DIFF
# ============================================================

# Effectifs de chaque classe
table(train$DIFF)

# Proportions de chaque classe
prop.table(table(train$DIFF))

# Graphique de la variable DIFF
barplot(table(train$DIFF),
        main = "Répartition de DIFF",
        xlab = "DIFF",
        ylab = "Effectif")


# ============================================================
# 5. ANALYSE DES VARIABLES EXPLICATIVES
# ============================================================

# Résumé statistique
summary(train)

# Histogrammes des variables numériques
hist(train$TOF,
     main = "Distribution de TOF",
     xlab = "TOF")

hist(train$AGE,
     main = "Distribution de AGE",
     xlab = "AGE")

hist(train$R7,
     main = "Distribution de R7",
     xlab = "R7")

hist(train$R8,
     main = "Distribution de R8",
     xlab = "R8")

hist(train$R17,
     main = "Distribution de R17",
     xlab = "R17")

hist(train$R22,
     main = "Distribution de R22",
     xlab = "R22")

hist(train$R32,
     main = "Distribution de R32",
     xlab = "R32")


# ============================================================
# 6. RELATION ENTRE LES VARIABLES ET DIFF
# ============================================================

# Comparer les distributions selon DIFF

boxplot(TOF ~ DIFF,
        data = train,
        main = "TOF selon DIFF",
        xlab = "DIFF",
        ylab = "TOF")

boxplot(AGE ~ DIFF,
        data = train,
        main = "AGE selon DIFF",
        xlab = "DIFF",
        ylab = "AGE")

boxplot(R7 ~ DIFF,
        data = train,
        main = "R7 selon DIFF",
        xlab = "DIFF",
        ylab = "R7")

boxplot(R8 ~ DIFF,
        data = train,
        main = "R8 selon DIFF",
        xlab = "DIFF",
        ylab = "R8")

boxplot(R17 ~ DIFF,
        data = train,
        main = "R17 selon DIFF",
        xlab = "DIFF",
        ylab = "R17")

boxplot(R22 ~ DIFF,
        data = train,
        main = "R22 selon DIFF",
        xlab = "DIFF",
        ylab = "R22")

boxplot(R32 ~ DIFF,
        data = train,
        main = "R32 selon DIFF",
        xlab = "DIFF",
        ylab = "R32")


# ============================================================
# 7. MATRICE DE CORRÉLATIONS
# ============================================================

# Corrélations entre variables quantitatives
cor(train[, colonnes_num], use = "complete.obs")

# Corrélation de chaque variable avec DIFF (0/1)
sapply(train[, colonnes_num], function(x) cor(x, as.numeric(as.character(train$DIFF))))


# ============================================================
# 8. VISUALISATION DES RELATIONS ENTRE VARIABLES
# ============================================================

pairs(train[, colonnes_num],
      col = ifelse(train$DIFF == "1", "orange", "blue"),
      pch = 19,
      main = "Relations entre les variables (orange = DIFF 1, bleu = DIFF 0)")
