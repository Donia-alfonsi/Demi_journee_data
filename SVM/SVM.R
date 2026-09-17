library(e1071)
library(pROC)

train <- read.csv2("../donnees_train.csv")
test  <- read.csv2("../donnees_test.csv")
#verif <- read.csv2("donnees_verification.csv")

str(train)

colonnes_num <- c("TOF", "AGE", "R7", "R8", "R17", "R22", "R32")

for (col in colonnes_num) {
  if (!is.numeric(train[[col]])) {
    train[[col]] <- as.numeric(gsub(",", ".", as.character(train[[col]])))
  }
  if (!is.numeric(test[[col]])) {
    test[[col]] <- as.numeric(gsub(",", ".", as.character(test[[col]])))
  }
}

train$DIFF <- as.factor(train$DIFF)
#verif$DIFF <- as.factor(verif$DIFF)

pairs(train[, colonnes_num],
      col = ifelse(train$DIFF == "1", "orange", "blue"),
      pch = 19)

C_sep <- c(0.001, 0.01, 1, 10, 100, 1000, 1e5)
nbMalClassees <- numeric(length(C_sep))

for (i in seq_along(C_sep)) {
  svm.tmp <- svm(DIFF ~ ., data = train, kernel = "linear", cost = C_sep[i])
  pred.tmp <- predict(svm.tmp, newdata = train)
  nbMalClassees[i] <- sum(pred.tmp != train$DIFF)
}

resultat_separabilite <- data.frame(C = C_sep, nb_mal_classees = nbMalClassees)
print(resultat_separabilite)

nbMalClassees_noscale <- numeric(length(C_sep))

for (i in seq_along(C_sep)) {
  svm.tmp <- svm(DIFF ~ ., data = train, kernel = "linear",
                 cost = C_sep[i], scale = FALSE)
  pred.tmp <- predict(svm.tmp, newdata = train)
  nbMalClassees_noscale[i] <- sum(pred.tmp != train$DIFF)
}

resultat_separabilite$nb_mal_classees_scaleFALSE <- nbMalClassees_noscale
print(resultat_separabilite)

age_mean <- mean(train$AGE)
age_sd   <- sd(train$AGE)

train$AGE <- (train$AGE - age_mean) / age_sd
test$AGE  <- (test$AGE  - age_mean) / age_sd

normaliser_cosinus <- function(data, colonnes, moyennes, ecarts_types) {
  X <- data[, colonnes]
  X <- sweep(X, 2, moyennes, "-")
  X <- sweep(X, 2, ecarts_types, "/")
  normes <- sqrt(rowSums(X^2))
  X <- X / normes
  data[, colonnes] <- X
  data
}

moyennes_train     <- sapply(train[, colonnes_num], mean)
ecarts_types_train <- sapply(train[, colonnes_num], sd)

train_cos <- normaliser_cosinus(train, colonnes_num, moyennes_train, ecarts_types_train)
test_cos  <- normaliser_cosinus(test,  colonnes_num, moyennes_train, ecarts_types_train)

set.seed(1)

tune.linear <- tune(
  svm, DIFF ~ ., data = train, kernel = "linear",
  ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)),
  tunecontrol = tune.control(cross = 10)
)

# noyau radial = noyau gaussien du cours, avec gamma = 1 / (2 * sigma^2)
sigma_radial <- c(7, 2, 0.7, 0.3)
gamma_radial <- 1 / (2 * sigma_radial^2)

tune.radial <- tune(
  svm, DIFF ~ ., data = train, kernel = "radial",
  ranges = list(cost = c(0.01, 0.1, 1, 10, 100), gamma = gamma_radial),
  tunecontrol = tune.control(cross = 10)
)

tune.poly <- tune(
  svm, DIFF ~ ., data = train, kernel = "polynomial",
  ranges = list(cost = c(0.01, 0.1, 1, 10), gamma = c(0.01, 0.1, 1),
                degree = c(2, 3, 4), coef0 = c(0, 1)),
  tunecontrol = tune.control(cross = 10)
)

tune.cosine <- tune(
  svm, DIFF ~ ., data = train_cos, kernel = "linear", scale = FALSE,
  ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)),
  tunecontrol = tune.control(cross = 10)
)

summary(tune.linear)
plot(tune.linear)
summary(tune.radial)
plot(tune.radial)
summary(tune.poly)
summary(tune.cosine)

entrainer_avec_proba <- function(tune.out, kernel) {
  best <- tune.out$best.model
  args <- list(formula = DIFF ~ ., data = train, kernel = kernel,
               cost = best$cost, probability = TRUE)
  if (!is.null(best$gamma))  args$gamma  <- best$gamma
  if (!is.null(best$degree)) args$degree <- best$degree
  if (!is.null(best$coef0))  args$coef0  <- best$coef0
  do.call(svm, args)
}

modele_linear <- entrainer_avec_proba(tune.linear, "linear")
modele_radial <- entrainer_avec_proba(tune.radial, "radial")
modele_poly   <- entrainer_avec_proba(tune.poly,   "polynomial")

modele_cosine <- svm(DIFF ~ ., data = train_cos, kernel = "linear",
                     scale = FALSE, cost = tune.cosine$best.model$cost,
                     probability = TRUE)

calculer_proba <- function(modele, newdata) {
  pred <- predict(modele, newdata = newdata, probability = TRUE)
  attr(pred, "probabilities")[, "1"]
}

proba_linear <- calculer_proba(modele_linear, test)
proba_radial <- calculer_proba(modele_radial, test)
proba_poly   <- calculer_proba(modele_poly,   test)
proba_cosine <- calculer_proba(modele_cosine, test_cos)

# le calcul des ROC/AUC a besoin des vraies valeurs de DIFF (verif) ;
# sur le test final "à l'aveugle" on n'a pas ça, donc on ne fait cette
# partie que si verif existe, pour ne pas planter le reste du script.
if (exists("verif")) {
  
  roc_linear <- roc(response = verif$DIFF, predictor = proba_linear)
  roc_radial <- roc(response = verif$DIFF, predictor = proba_radial)
  roc_poly   <- roc(response = verif$DIFF, predictor = proba_poly)
  roc_cosine <- roc(response = verif$DIFF, predictor = proba_cosine)
  
  auc(roc_linear)
  auc(roc_radial)
  auc(roc_poly)
  auc(roc_cosine)
  
  par(mfrow = c(2, 2))
  plot(roc_linear, main = paste("linear - AUC =", round(auc(roc_linear), 3)))
  plot(roc_radial, main = paste("radial (gaussien) - AUC =", round(auc(roc_radial), 3)))
  plot(roc_poly, main = paste("polynomial - AUC =", round(auc(roc_poly), 3)))
  plot(roc_cosine, main = paste("cosinus - AUC =", round(auc(roc_cosine), 3)))
  par(mfrow = c(1, 1))
  
} else {
  cat("Pas de 'verif' disponible : on saute le calcul des ROC/AUC et on\n",
      "passe directement à l'export des prédictions finales.\n")
}

# à changer selon la courbe ROC qui vous semble la meilleure :
# "linear", "radial", "polynomial" ou "cosinus"
meilleur_noyau <- "radial"

if (meilleur_noyau == "cosinus") {
  modele_final  <- modele_cosine
  newdata_final <- test_cos
} else {
  modele_final <- switch(meilleur_noyau,
                         linear     = modele_linear,
                         radial     = modele_radial,
                         polynomial = modele_poly)
  newdata_final <- test
}

pred_final <- predict(modele_final, newdata = newdata_final)

id_test <- if (!is.null(test$ID)) test$ID else seq_len(nrow(test))

soumission <- data.frame(ID = id_test, DIFF = as.integer(as.character(pred_final)))
write.csv(soumission, "soumission.csv", row.names = FALSE, quote = FALSE)