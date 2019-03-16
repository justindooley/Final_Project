# k-means is an unsupervised machine learning algorithm 
# used to find groups of observations (clusters) 
# that share similar characteristics.

# Load libraries

library(tidyverse)
library(dplyr)
library(tidyr)
library(readr)
library(tibble)
library(corrplot)
library(gridExtra)
library(GGally)
library(stats)
library(ggplot2)
library(fpc)
library(cluster)

# Set the seed for reproducibility
set.seed(1002476)

# Set the working directory
getwd()
path.expand("~/kmeans_analysis")
setwd("~/kmeans_analysis")


# Read the stats
wines <- read.csv("input/combined_wine_binary.csv", header = TRUE)

# View header
head(wines, n=10) 

# Rename the columns
names(wines) <- c("wine_type", "fixed_acidity", "volatile_acidity", "citric acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality")

## DATA VISUALIZATION AND DATA EXPLORATION

# Determine the structure of data set
str(wines)

# Examine the statistical distribution
summary(wines)

# Standardize the variables
wines <- scale(wines) 

# Create a histogram for each attribute
wines %>%
  gather(Attributes, value, 1:13) %>%
  ggplot(aes(x=value)) +
  geom_histogram(fill="purple", colour="black") +
  facet_wrap(~Attributes, scales="free_x") +
  labs(x="Values", y="Frequency") 

# Create a correlation matrix for each attribute
corrplot(cor(wines), type="upper", method="ellipse", tl.cex=0.9)

# Graph Scatter of Data
# Type versus Sulfur Dioxide
ggplot(data = wines) + 
  geom_point(mapping = aes(x = total_sulfur_dioxide, y = alcohol))

# Pair-Wise Correlation 
ggpairs(cbind(wines, Cluster=as.factor(wines$quality)),
        columns=1:13, aes(colour=Cluster, alpha=0.5),
        lower=list(continuous="points"),
        upper=list(continuous="blank"),
        axisLabels="none", switch="both")

# Determine the best k-means value 

bss <- numeric()
wss <- numeric()

for(i in 1:13) 
  
{
  # For each k, calculate betweenss and tot.withinss
  bss[i] <- kmeans(wines, centers=i)$betweenss
  wss[i] <- kmeans(wines, centers=i)$tot.withinss
}

# To study which value of k gives us the best partition
# Between-Cluster Sum of Squares vs Choice of k
p3 <- qplot(1:13, bss, geom=c("point", "line"), 
            xlab="Number of Clusters", ylab="Between-Cluster Sum of Squares") +
  scale_x_continuous(breaks=seq(0, 13, 1))

# Total within-cluster sum of squares vs Choice of k
p4 <- qplot(1:13, wss, geom=c("point", "line"),
            xlab="Number of Clusters", ylab="Total Within-Cluster Sum of Squares") +
  scale_x_continuous(breaks=seq(0, 13, 1))

# Subplot
grid.arrange(p3, p4, ncol=2)

# Determine the optimal k-value based on graph
# Choose number of clusters where adding 
# another cluster doesn’t partition data better
# Choose k-value of 5

# Testing and execution of k-means with k=10
wines_10_centers <-kmeans(wines, centers=10) 
wines_10_centers$centers # Display cluster centers
table(wines_10_centers$cluster) # Give a count of data points in each cluster


wines_10 <-kmeans(wines[,-c(13)], centers=10) 
wines_10$centers # Display cluster centers
table(wines_10$cluster) # Give a count of data points in each cluster
table(wines_10_centers$cluster)

# Mean values of each cluster
mean_value <- aggregate(wines, by=list(wines_10$cluster), mean)
mean_value
summary(mean_value)

## K-Means Execution for:
# 2 (red or white wine)
# 5 (best partition)
# 10 (quality)

# K-Means Cluster Analysis with k=2
fit <- kmeans(wines, 2) # 2 cluster solution
aggregate(wines, by=list(fit$cluster), FUN=mean) # get cluster means 
wines_k2 <- data.frame(wines, fit$cluster) # append character assignment

# Write results to a new CSV file to visualize in Tableau 
write.csv(wines_k2, file = "wines_k2.csv")

# K-Means Cluster Analysis with k=5
fit <- kmeans(wines, 5) # 2 cluster solution
aggregate(wines, by=list(fit$cluster), FUN=mean) # get cluster means 
wines_k5 <- data.frame(wines, fit$cluster) # append character assignment

# Write results to a new CSV file to visualize in Tableau 
write.csv(wines_k5, file = "wines_k5.csv")

# K-Means Cluster Analysis with k=10
fit <- kmeans(wines, 10) # 2 cluster solution
aggregate(wines, by=list(fit$cluster), FUN=mean) # get cluster means 
wines_k10 <- data.frame(wines, fit$cluster) # append character assignment

# Write results to a new CSV file to visualize in Tableau 
write.csv(wines_k10, file = "wines_10.csv")



