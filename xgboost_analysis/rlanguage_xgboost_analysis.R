# Extreme Gradient Boosting is a supervised machine learning model with
# both a linear model solver and tree learning algorithms. 
# So, what makes it fast is its capacity to do 
# parallel computation on a single machine.

# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html
# https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
# https://www.analyticsvidhya.com/blog/2016/02/complete-tutorial-learn-data-science-scratch/#four

# Load all libraries you could ever want or need
# just to be safe rather than sorry. 

library(data.table)
library(tidyverse)
library(dplyr)
library(tidyr)
library(readr)
library(tibble)
library(corrplot)
library(gridExtra)
library(GGally)
library(stats)
library(xgboost)
library(stringr)
library(caret)
library(mlr)

# Set the working directory
getwd()
path.expand("~/kmeans_analysis")
setwd()

# Read the stats
wines <- read.csv("input/combined_wine_binary.csv", header = TRUE)

# View header
head(wines, n=10) 

# Rename the columns
names(wines) <- c("wine_type", "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality")

## DATA VISUALIZATION AND DATA EXPLORATION

# Determine the structure of data set
str(wines)

# Examine the statistical distribution
summary(wines)

# Create a histogram for each attribute
wines %>%
  gather(Attributes, value, 1:13) %>%
  ggplot(aes(x=value)) +
  geom_histogram(fill="purple", colour="black") +
  facet_wrap(~Attributes, scales="free_x") +
  labs(x="Values", y="Frequency") 

# Create a correlation matrix for each attribute
corrplot(cor(wines), type="upper", method="ellipse", tl.cex=0.9)

## XGBOOST ANALYSIS, LET'S GO 

# Partition the data into train and test sets (80/20)
sample_size <- floor(0.80 * nrow(wines))

# Set the seed to make the partition reproducible
set.seed(2476)
train_index <- sample(seq_len(nrow(wines)), size = sample_size)

train <- wines[train_index, ]
test <- wines[-train_index, ]

# Convert dataframes to data tables
setDT(train) 
setDT(test)

# Identify the label using one hot encoding 
labels <- train$wine_type
ts_label <- test$wine_type

# Should I train the model to wine_type (a binary) or 
# quality, which I would need to one-hot encode?
new_tr <- model.matrix(~.+0,data = train[,-c("wine_type"), with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("wine_type"), with=F])

# Convert data table into a matrix using xgb.DMatrix
dtrain <- xgb.DMatrix(data = new_tr, label = labels) 
dtest <- xgb.DMatrix(data = new_ts, label = ts_label)

# Set the default parameters 
params <- list(booster = "gbtree", 
            objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, 
            min_child_weight=1, subsample=1, colsample_bytree=1)

# Find the best iteration of the model 
xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 1000, 
        nfold = 5, showsd = T, stratified = T,
        print_every_n = 10, early_stopping_rounds = 20, maximize = F)

# Best Round for 1000 = [61]

# Find test error mean
## Error: no non-missing arguments to min; returning Inf
min(xgbcv$test.error.mean)

# CV accuracy (100 - test error mean) = XX.XX% 

# First Default - Model Training

xgb1 <- xgb.train(params = params, data = dtrain, nrounds = 61, 
                  watchlist = list(val=dtest, train=dtrain), 
                  print_every_n = 10, early_stop_round = 10, 
                  maximize = F, eval_metric = "error")

# Model prediction

xgbpred <- predict (xgb1, dtest)
xgbpred <- ifelse (xgbpred > 0.5, 1, 0)

# Calculate model's accuracy using Caret's confusionMatrix
## Error: `data` and `reference` should be factors with the same levels.
confusionMatrix(xgbpred, new_ts)

# view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr), model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20]) 