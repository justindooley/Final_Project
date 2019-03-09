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

# Set the working directory
getwd()
setwd("C:/Users/Courtney/Documents/wine_datasets/input/")

# Read the stats
wines <- read.csv("combined_wine_binary.csv", header = TRUE)

# View header
head(wines) 

# DATA ANALYSIS: VISUALIZATION AND DATA EXPLORATION

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

# Since there's a strong linear correlation
# between sulfur dioxide and type, 
# test the relationship between them 

# Relationship between 
# sulfur dioxide and wine type
ggplot(wines, aes(x=total.sulfur.dioxide, y=Type)) +
  geom_point() +
  geom_smooth(method="lm", se=FALSE)

# K-Means Execution

# Execution of k-means with k=2
set.seed(1234)
wines_k2 <- kmeans(wines, centers=2)

# Cluster to which each point is allocated
wines_k2$cluster

# Cluster centers
wines_k2$centers

# Cluster sizes
wines_k2$size

# Between-cluster sum of squares
wines_k2$betweenss

# Within-cluster sum of squares
wines_k2$withinss

# Total within-cluster sum of squares 
wines_k2$tot.withinss

# Total sum of squares
wines_k2$totss

# Run the algorithm for different values of k 
bss <- numeric()
wss <- numeric()

set.seed(1234)
for(i in 1:10){
  
  # For each k, calculate betweenss and tot.withinss
  bss[i] <- kmeans(winesNorm, centers=i)$betweenss
  wss[i] <- kmeans(winesNorm, centers=i)$tot.withinss
}

# Between-cluster sum of squares vs Choice of k
p3 <- qplot(1:10, bss, geom=c("point", "line"), 
            xlab="Number of Clusters", ylab="Between-Cluster Sum of Squares") +
  scale_x_continuous(breaks=seq(0, 10, 1))

# Total within-cluster sum of squares vs Choice of k
p4 <- qplot(1:10, wss, geom=c("point", "line"),
            xlab="Number of Clusters", ylab="Total Within-Cluster Sum of Squares") +
  scale_x_continuous(breaks=seq(0, 10, 1))

# Subplot
grid.arrange(p3, p4, ncol=2)

## RESULTS

# Execution of k-means with k=3
set.seed(1234)
wines_k2 <- kmeans(wines, centers=3)

# Mean values of each cluster
aggregate(wines, by=list(wines_k2$cluster), mean)

# Clustering 
ggpairs(cbind(wines, Cluster=as.factor(wines_k2$cluster)),
        columns=1:13, aes(colour=Cluster, alpha=0.5),
        lower=list(continuous="points"),
        upper=list(continuous="blank"),
        axisLabels="none", switch="both")

