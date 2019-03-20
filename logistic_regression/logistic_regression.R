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

factor_wines <- factor(raw_data$quality, labels = c("less than 5", 
                                          "greater than or equal to 5"))
factor_wines

wines$quality <- factor(wines$quality)

# Check if the variables are factors
is.factor(factor_wines)
contrasts(factor_wines)

# Split the data into train and test sets using the caret package
# Partition the data into train and test sets (80/20)
sample_size <- floor(0.80 * nrow(raw_data))

train_index <- sample(seq_len(nrow(raw_data)), size = sample_size)

train <- raw_data[train_index, ]
test <- raw_data[-train_index, ]

# Build the logistic regression model

model <- glm(quality ~ wine_type + fixed_acidity + volatile_acidity 
             + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide 
             + total_sulfur_dioxide + density + pH + sulphates 
             + alcohol, data = train, family = binomial(link="logit"))

# Predicted scores
predicted <- plogis(predict(model, test)) 

# Determine the optimal cut-off point
optCutOff <- optimalCutoff(test$quality, predicted)[1]
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

fitted.results <- predict(model, newdata = subset(test), type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != test$quality)
print(paste('Accuracy', 1-misClasificError))

# Accuracy = 0.960769230769231 = 96.54%

# Plot the ROC and AUC
plotROC(test$quality, predicted)
plotAUC(test$quality, predicted)

# Of all combinations of 1-0 pairs (actuals), concordance is the percentage
# of pairs, whose scores of positives are greater than the scores of negatives. 
# For a perfect model, this will be 100%. 
# The higher the concordance, the better is the quality of model.
Concordance(test$quality, predicted)

# Concordance = [1] 0.7348832 = ~73%
# Not that great of a fit 

sensitivity(test$quality, predicted, threshold = optCutOff)
# [1] 0.9984051

specificity(test$quality, predicted, threshold = optCutOff)
# [1] 0.1086957

confusionMatrix(test$quality, predicted, threshold = optCutOff)
# The columns are actuals, while rows are predicted values

#    0    1
# 0  5    2
# 1 41 1252

# Another way of calculating ROC and AUC

ROC1 <- roc(test$quality, predicted)

plot(ROC1, col = "blue")

AUC1 <- auc(ROC1)

# Area under the curve: 0.7107
