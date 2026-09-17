
train <- read.csv2("donnees_train.csv")
validation <- read.csv2("donnees_verification.csv")
test <- read.csv2("donnees_test.csv")

head(train)
dim(train)
dim(validation)
summary(train)

# Vérification
str(train)

# Regarder si des valeurs posent problème
sapply(train, function(x) {
  if (is.character(x)) sum(is.na(suppressWarnings(as.numeric(gsub(",", ".", x)))))
  else NA
})

# Convertir en numérique
train[] <- lapply(train, function(x) {
  if (is.character(x)) as.numeric(gsub(",", ".", x)) else x
})

validation[] <- lapply(validation, function(x) {
  if (is.character(x)) as.numeric(gsub(",", ".", x)) else x
})

test[] <- lapply(test, function(x) {
  if (is.character(x)) as.numeric(gsub(",", ".", x)) else x
})

summary(train)

# Normalisation (DIFF mise de côté avant scale, remise après)

train_diff <- train$DIFF
train <- as.data.frame(scale(train))
train$DIFF <- train_diff

validation_diff <- validation$DIFF
validation <- as.data.frame(scale(validation))
validation$DIFF <- validation_diff

test <- as.data.frame(scale(test))

head(train)
summary(train)


# Premières visualisations (ici on remplace la notationn 0, 1 par 1, 2)

pairs(train[1:8], main = "Données Demi journée data 2 groupes", pch = 21, bg = c("blue", "orange")[as.factor(train$DIFF)], lower.panel=NULL, font.labels=0.5, cex.labels=2) 
legend(0.08, 0.43, as.vector(unique(train$DIFF)),  fill=c("blue", "orange"))


# Proportion d'individu

round(prop.table(table(train$DIFF)) * 100, digits = 1)


# -----------------

# Début modèle KNN : recherche de K

library(class)
library(pROC)

K.max <- 50
auc.values <- numeric(K.max)

for (i in 1:K.max) {
  knn.pred <- knn(train[, -1], validation[, -1], train[, "DIFF"], k = i, prob = TRUE)
  
  # Proportion gagnante
  prob.gagnante <- attr(knn.pred, "prob")
  
  # Probabilité DIFF = 1
  prob.classe1 <- ifelse(knn.pred == 1, prob.gagnante, 1 - prob.gagnante)
  
  roc.obj <- roc(validation[, "DIFF"], prob.classe1, quiet = TRUE)
  auc.values[i] <- auc(roc.obj)
}

plot(auc.values, type = "b", col = "dodgerblue", pch = 20,
     xlab = "K, nombre de voisins", ylab = "AUC (validation)",
     main = "AUC vs Nombre de voisins")
abline(v = which(auc.values == max(auc.values)), col = "darkorange", lwd = 1.5)

meilleur.K <- which(auc.values == max(auc.values))
meilleur.K
max(auc.values)

# ------------------------
# Modèle final avec K = 28

knn.pred.final <- knn(train[, -1], validation[, -1], train[, "DIFF"], k = meilleur.K, prob = TRUE)

prob.gagnante <- attr(knn.pred.final, "prob")
prob.classe1  <- ifelse(knn.pred.final == 1, prob.gagnante, 1 - prob.gagnante)

# Matrice de confusion
table(validation[, "DIFF"], knn.pred.final)

# Accuracy
mean(validation[, "DIFF"] != knn.pred.final)
mean(validation[, "DIFF"] == knn.pred.final) * 100

# Courbe ROC et AUC
roc.obj.final <- roc(validation[, "DIFF"], prob.classe1, quiet = TRUE)
plot(roc.obj.final, main = paste("Courbe ROC - K =", meilleur.K))
auc(roc.obj.final)

# ------------------------
# Prédiction sur le jeu de données test

head(test)
dim(test)
str(test)
summary(test)


# Prédiction

knn.pred.test <- knn(
  train[, -1],
  test,
  train[, "DIFF"],
  k = meilleur.K,
  prob = TRUE
)

knn.pred.test

# Création fichier

resultats.test <- data.frame(
  ID = 1:120,
  DIFF = knn.pred.test
)

write.csv2(
  resultats.test,
  "predictions_KNN.csv",
  row.names = FALSE
)
