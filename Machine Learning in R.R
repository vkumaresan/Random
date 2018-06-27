# Machine Learning in R
head(mtcars)
pairs(mtcars[1:7], lower.panel = NULL)

plot(y = mtcars$mpg, x = mtcars$wt, xlab = "Vehicle Weight",    ylab = "Vehicle Fuel Efficiency in Miles per Gallon")

mt.model <- lm(formula = mpg ~ wt, data = mtcars)
coef(mt.model)[2]
coef(mt.model)[1]

# Regression

model <- lm(mtcars$mpg ~ mtcars$disp)
coef(model)
coef(model)[2] * 200 + coef(model)[1]

summary(model)

split_size = 0.8

sample_size = floor(split_size * nrow(mtcars))

set.seed(123)
train_indices <- sample(seq_len(nrow(mtcars)), size = sample_size)

train <- mtcars[train_indices, ]
test <- mtcars[-train_indices, ]

model2 <- lm(mpg ~ disp, data = train)

new.data <- data.frame(disp = test$disp)

test$output <- predict(model2, new.data)

sqrt(sum(test$mpg - test$output)^2/nrow(test))

# Logistic Regression
library(caTools)

Label.train = train[,9]
Data.train = train[,-9]

model = LogitBoost(Data.train, Label.train)
Data.test = test
Lab = predict(model, Data.test, type = "raw")
data.frame(row.names(test), test$mpg, test$am, Lab)

# Supervised Clustering
data = data.frame(iris$Petal.Length, iris$Petal.Width)

iris.kmeans <- kmeans(data, 2)
plot(x = iris$Petal.Length, y = iris$Petal.Width, pch = iris.kmeans$cluster,
     xlab = "Petal Length", ylab = "Petal Width")
points(iris.kmeans$centers, pch = 8, cex = 2)

iris.kmeans3 <- kmeans(data, 3)

plot(x = iris$Petal.Length, y = iris$Petal.Width, pch = iris.kmeans3$cluster,
     xlab = "Petal Length", ylab = "Petal Width")

points(iris.kmeans3$centers, pch = 8, cex = 2)

par(mfrow = c(1,2))

plot(x = iris$Petal.Length, y = iris$Petal.Width, pch = iris.kmeans3$cluster,
     xlab = "Petal Length", ylab = "Petal Width", main = "Model Output")

plot(x = iris$Petal.Length, y = iris$Petal.Width,
     pch = as.integer(iris$Species),
     xlab = "Petal Length", ylab = "Petal Width", main = "Actual Data")
table(iris.kmeans3$cluster, iris$Species)

# Mixed Models
# Tree-Based Models
library(party)
tree <- ctree(mpg ~., data = mtcars)
plot(tree)

tree.train <- ctree(mpg ~ ., data = train)
plot(tree.train)

test$mpg.tree <- predict(tree.train, test)
test$class <- predict(tree.train, test, type = "node")
data.frame(row.names(test), test$mpg, test$mpg.tree, test$class)

# Random Forests
library(randomForest)

mtcars.rf <- randomForest(mpg ~ ., data = mtcars, ntree = 1000, keep.forest = FALSE, importance = FALSE)

plot(mtcars.rf, log = "y", title = "")

# Neural Network

set.seed(123)
library(nnet)
iris.nn <- nnet(Species ~ ., data = iris, size = 2)
table(iris$Species, predict(iris.nn, iris, type = "class"))

# Support Vector Machines
library(e1071)
iris.svm <- svm(Species ~ ., data = iris)
table(iris$Species, predict(iris.svm, iris, type = "class"))

# Unsupervised Clustering
x <- rbind(matrix(rnorm(100, sd = 0.3), ncol = 2), matrix(rnorm(100, mean = 1, sd = 0.3), ncol = 2))
colnames(x) <- c("x", "y")
plot(x)
cl <- kmeans(x, 2)
plot(x, pch=cl$cluster)
cl[2]

# Sampling
iris.df <- data.frame(iris)
sample.index <- sample(1:nrow(iris.df), nrow(iris) * 0.75, replace = FALSE)
head(iris[sample.index, ])
summary(iris)
summary(iris[sample.index, ])

sys.sample = function(N,n) {
  k = ceiling(N/n)
  r = sample(1:k, 1)
  sys.samp = seq(r, r+k * (n-1), k)
}

systematic.index <- sys.sample(nrow(iris), nrow(iris) * 0.75)
summary(iris[systematic.index, ])

# Training and Test Sets: Regression Modeling
set.seed(123)
x <- rnorm(100,2,1)
y = exp(x) + rnorm(5, 0, 2)
plot(x, y)
linear <- lm(y ~ x)
abline(a = coef(linear[1], b = coef(linear[2], lty = 2)))

data <- data.frame(x, y)
data.samples <- sample(1:nrow(data), nrow(data) * 0.7, replace = FALSE)
training.data <- data[data.samples, ]
test.data <- data[-data.samples, ]

train.linear <- lm(y ~ x, training.data)
train.output <- predict(train.linear, test.data)
RMSE.df = data.frame(predicted = train.output, actual = test.data$y,
                     SE = ((train.output - test.data$y)^2/length(train.output)))
head(RMSE.df)
sqrt(sum(RMSE.df$SE))

train.quadratic <- lm(y ~ x^2 + x, training.data)
quadratic.output <- predict(train.quadratic, test.data)
RMSE.quad.df = data.frame(predicted = quadratic.output, actual = test.data$y,
                          SE = ((quadratic.output - test.data$y)^2/length(train.output)))
head(RMSE.quad.df)
sqrt(sum(RMSE.quad.df$SE))

iris.df <- iris
iris.df$Species <- as.character(iris.df$Species)
iris.df$Species[iris.df$Species != "setosa"] <- "other"
iris.df$Species <- as.factor(iris.df$Species)
iris.samples <- sample(1:nrow(iris.df), nrow(iris.df) * 0.7, replace = FALSE)
training.iris <- iris.df[iris.samples, ]
test.iris <- iris.df[-iris.samples, ]
library(randomForest)
iris.rf <- randomForest(Species ~ ., data = training.iris)
iris.predictions <- predict(iris.rf, test.iris)
table(iris.predictions, test.iris$Species)

# k-fold cross-validation
set.seed(123)

x <- rnorm(100,2,1)
y = exp(x) + rnorm(5,0,2)
data <- data.frame(x, y)

data.shuffled <- data[sample(nrow(data)), ]
folds <- cut(seq(1, nrow(data)), breaks = 10, labels = FALSE)

errors <- c(0)

for (i in 1:10) {
  fold.indexes <- which(folds == i, arr.ind = TRUE)
  
  test.data <- data[fold.indexes, ]
  training.data <- data[-fold.indexes, ]
  
  train.linear <- lm(y~x, training.data)
  train.output <- predict(train.linear, test.data)
  errors <- c(errors, sqrt(sum(((train.output - test.data$y)^2/length(train.output)))))
}

errors[2:11]
mean(errors[2:11])

# lasso regression
library(lasso2)
lm.lasso <- l1ce(mpg ~ ., data = mtcars)
summary(lm.lasso)$coefficients

# logistic regression
iris.binary <- iris
iris.binary$binary <- as.numeric(iris[, 5] == "setosa")


iris.logistic <- glm(binary ~ Sepal.Width + Sepal.Length, data = iris.binary,
                     family = "binomial")
iris.logistic

library(caret)

data("GermanCredit")

Train <- createDataPartition(GermanCredit$Class, p = 0.6, list = FALSE)
training <- GermanCredit[Train, ]
testing <- GermanCredit[-Train, ]

mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate +
                   Housing.Own + CreditHistory.Critical, data = training, method = "glm",
                 family = "binomial")


predictions <- predict(mod_fit, testing[, -10])
table(predictions, testing[, 10])

# change logistic regression algorithm
mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate +
                   Housing.Own + CreditHistory.Critical, data = training,
                 method = "LogitBoost",
                 family = "binomial")

predictions <- predict(mod_fit, testing[, -10])
table(predictions, testing[, 10])

# Neural Networks
library(neuralnet)

set.seed(123)
AND <- c(rep(0, 3), 1)
binary.data <- data.frame(expand.grid(c(0, 1), c(0, 1)), AND)
net <- neuralnet(AND ~ Var1 + Var2, binary.data, hidden = 0,
                 err.fct = "ce", linear.output = FALSE)
plot(net, rep = "best")

prediction(net)

set.seed(123)
AND <- c(rep(0, 7), 1)
OR <- c(0, rep(1, 7))
binary.data <- data.frame(expand.grid(c(0, 1), c(0, 1), c(0,
                                                          1)), AND, OR)
net <- neuralnet(AND + OR ~ Var1 + Var2 + Var3, binary.data,
                 hidden = 0, err.fct = "ce", linear.output = FALSE)
plot(net, rep = "best")
prediction(net)

set.seed(123)
AND <- c(rep(0, 7), 1)
binary.data <- data.frame(expand.grid(c(0, 1), c(0, 1), c(0,
                                                          1)), AND, OR)
net <- neuralnet(AND ~ Var1 + Var2 + Var3, binary.data, hidden = 1,
                 err.fct = "ce", linear.output = FALSE)
plot(net, rep = "best")
 # NN for Regression
library(mlbench)
data(BostonHousing)

lm.fit <- lm(medv ~ ., data = BostonHousing)

lm.predict <- predict(lm.fit)
library(nnet)

nnet.fit1 <- nnet(medv/50 ~ ., data = BostonHousing, size = 2, maxit = 1000, trace = FALSE)
nnet.predict1 <- predict(nnet.fit1) * 50

plot(BostonHousing$medv, nnet.predict1, main = "Neural network predictions vs
    actual with normalized response inputs",
     xlab = "Actual", ylab = "Prediction")
library(caret)

mygrid <- expand.grid(.decay = c(0.5, 0.1), .size = c(4, 5, 6))
nnetfit <- train(medv/50 ~ ., data = BostonHousing, method = "nnet",
                 maxit = 1000, tuneGrid = mygrid, trace = F)
print(nnetfit)

# NN for Classification
iris.df <- iris
smp_size <- floor(0.75 * nrow(iris.df))

set.seed(123)
train_ind <- sample(seq_len(nrow(iris.df)), size = smp_size)

train <- iris.df[train_ind, ]
test <- iris.df[-train_ind, ]

iris.nnet <- nnet(Species ~ ., data = train, size = 4, decay = 0.0001,
                  maxit = 500, trace = FALSE)
predictions <- predict(iris.nnet, test[, 1:4], type = "class")
table(predictions, test$Species)

# NN w/ Caret
library(car)
library(caret)
trainIndex <- createDataPartition(Prestige$income, p = 0.7, list = F)
prestige.train <- Prestige[trainIndex, ]
prestige.test <- Prestige[-trainIndex, ]


my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6,
                                                       7))
prestige.fit <- train(income ~ prestige + education, data = prestige.train,
                      method = "nnet", maxit = 1000, tuneGrid = my.grid, trace = F,
                      linout = 1)

prestige.predict <- predict(prestige.fit, newdata = prestige.test)

summary(prestige.test$income)


iris.caret <- train(Species ~ ., data = train, method = "nnet",
                    trace = FALSE)
predictions <- predict(iris.caret, test[, 1:4])
table(predictions, test$Species)

library(rpart)
fit <- rpart(
  Mileage~Price + Country + Reliability + Type,
  method="anova", #method="class" for classificaiton tree
  data=cu.summary
)

plot(fit, uniform=TRUE, margin=0.1)
text(fit, use.n=TRUE, all=TRUE, cex=.8)
