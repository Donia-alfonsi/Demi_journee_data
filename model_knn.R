
data <- read.csv("farms_train.csv")
head(data)
dim(data)

summary(data)

X_scaled <- scale(data)
head(data)


plot(data)

table(data$DIFF, data$TOF)

pairs(data[1:8], main = "Données Demi journée data 2 groupes", pch = 21, bg = c("blue","orange")[data$DIFF], lower.panel=NULL, font.labels=0.5, cex.labels=2) 
legend(0.08, 0.43, as.vector(unique(iris$Species)),  fill=c("blue","orange", "green3"))
