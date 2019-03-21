# Load the libraries

library(tidyverse)
library(dplyr)
library(tidyr)
library(readr)
library(tibble)
library(stats)
library(ggplot2)
library(mltools)
library(data.table)
library(caret)
library(aod)
library(ROCR)
library(InformationValue)
library(pROC)

# Set the seed
set.seed(1002476)

# Load the data
raw_data <- read.csv("input/combined_wine_binary_quality.csv", header = TRUE)

# Rename the columns
names(raw_data) <- c("wine_type", "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality")

# View header
head(raw_data, n=10)

# Since logistic regression requires a binomial, translate the quality 
# column into two factors: quality scores below (0) and above (1) a value of 
# the median quality score (5). 

raw_data$quality <- factor(raw_data$quality, levels = c(0, 1))

# Check if the variables are factors
is.factor(raw_data$quality)
# [1] TRUE

# Check quality sampling imbalance
table(raw_data$quality)

#    0    1 
#   246 6251 

# First split the data into train and test sets using the caret package
# Partition the data into train and test sets (70/30)

'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.

# Prep Training and Test data.
trainDataIndex <- createDataPartition(raw_data$quality, p=0.7, list = F)  # 70% training data
trainData <- raw_data[trainDataIndex, ]
testData <- raw_data[-trainDataIndex, ]

table(trainData$quality)
#    0    1 
#    173 4376 

# In order to adjust for the high amount of mid-tier wines, we need to 
# upsample the data. 

# Up Sampling Code
up_train <- upSample(x = trainData[, colnames(trainData) %ni% "quality"],
                     y = trainData$quality)

# Check to see the new numbers
table(up_train$Class)

# 0    1 
# 4376 4376 


# Build the logistic regression model

model <- glm(Class ~ wine_type + fixed_acidity + volatile_acidity 
             + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide 
             + total_sulfur_dioxide + density + pH + sulphates 
             + alcohol, data = up_train, family = binomial(link="logit"))

# Predicted scores
predicted <- plogis(predict(model, testData)) 

# Determine the optimal cut-off point
optCutOff <- optimalCutoff(test$Class, predicted)[1]
# optCutOff = 0.6799987

# Summarize the findings

summary(model)

# Check for multicollinearity in the model
vif(model)

misClassError(test$quality, predicted, threshold = optCutOff)
#  0.0377

# Run an ANOVA
anova(model, test="Chisq")

## CIs using profiled log-likelihood
confint(model)

## CIs using standard errors
confint.default(model)

# Test for an overall effect of rank using the wald.test function of the aod 
# library. The order in which the coefficients are given in the 
# table of coefficients is the same as the order of the terms in the model.

wald.test(b = coef(model), Sigma = vcov(model), Terms = 2:13)

# Chi-squared test:
# X2 = 249.5, df = 12, P(> X2) = 0.0

# Above, we briefly evaluated the fitting of the model, 
# now test model by predicting y on a new set of data. 

fitted.results <- predict(model, newdata = subset(testData), type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != testData$quality)
print(paste('Accuracy', 1-misClasificError))

# Accuracy 0.756160164271047 = 75.61%

# Plot the ROC and AUC
plotROC(testData$quality, predicted)
plotAUC(testData$quality, predicted)

# Of all combinations of 1-0 pairs (actuals), concordance is the percentage
# of pairs, whose scores of positives are greater than the scores of negatives. 
# For a perfect model, this will be 100%. 
# The higher the concordance, the better is the quality of model.
Concordance(testData$quality, predicted)

# Concordance = [1] 0.7566393 = ~75%
# Not that great of a fit 

sensitivity(testData$quality, predicted)
# 76%

specificity(testData$quality, predicted)
# [1] 0.6575342 = 65.75%

confusionMatrix(testData$quality, predicted)
# The columns are actuals, while rows are predicted values

#    0    1
# 0 48  450
# 1 25 1425

# Another way of calculating ROC and AUC

ROC1 <- roc(testData$quality, predicted)

plot(ROC1, col = "blue")

AUC1 <- auc(ROC1)

# Area under the curve: 0.7566
