# SVM using Heart Disease Dataset 30-Aug-2020

#### Exploratory Data Analysis ####
library(caret)
heart <- read.csv("D:/4th trim/Machine Learning/Abhinav's Repository/SVM using Heart dataset/Heart Disease data set.csv", sep = ",", header = TRUE)
str(heart)

head(heart)

#### Dividing Dataset into Training and Testing ####
set.seed(3033)
intrain <- createDataPartition(y = heart$target, p = 0.7, list = FALSE)
training <- heart[intrain,] 
testing <- heart[-intrain,]

#### Checking Dimensions of Training & Testing dataframes ####
dim(training);
dim(testing);

#### Cleaning the data ####
anyNA(heart)

summary(heart)

#### Factorising Data to make the variable categorical ####
training[["target"]] = factor(training[["target"]])

#### Improving the estimated performance of the model ####
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#### Training the model ####
svm_Linear <- train(target ~., data = training, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

svm_Linear

#### Predicting classes for the test set ####
test_pred <- predict(svm_Linear, newdata = testing)
test_pred

#### Using the confusion matrix to predict the accuracy ####
confusionMatrix(table(test_pred, testing$target))

#### Improving the performance of the model ####
grid <- expand.grid(C = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5))
svm_Linear_Grid <- train(target ~., data = training, method = "svmLinear",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)

svm_Linear_Grid

#### Plotting the trained model ####
plot(svm_Linear_Grid)

#### Predicting classes for the test set again ####
test_pred_grid <- predict(svm_Linear_Grid, newdata = testing)
test_pred_grid

#### Using the confusion matrix to predict the accuracy of the improved model ####
confusionMatrix(table(test_pred_grid, testing$target))

#### Final accuracy of the improved model is 82.22% ####
