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
path.expand("~/kmeans_analysis")
setwd()


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

# Normalize the data so we get a standard deviation of 1
# and a mean of 0 
scaled_wines <- scale(wines)

# See what man hath wrot 
colMeans(scaled_wines)  
apply(scaled_wines, 2, sd)
head(scaled_wines, n=10)

# Create a histogram for each attribute 
wines %>%
  gather(Attributes, value, 1:13) %>%
  ggplot(aes(x=value)) +
  geom_histogram(fill="purple", colour="black") +
  facet_wrap(~Attributes, scales="free_x") +
  labs(x="Values", y="Frequency") 

# Somewhere, mistakes were made
# scaled_wines is not a dataframe somehow
# scaled_wines %>%
#  gather(Attributes, value, 1:13) %>%
#  ggplot(aes(x=value)) +
#  geom_histogram(fill="purple", colour="black") +
#  facet_wrap(~Attributes, scales="free_x") +
#  labs(x="Values", y="Frequency") 

# Create a correlation matrix for each attribute
corrplot(cor(wines), type="upper", method="ellipse", tl.cex=0.9)

## K-Means Execution

# Execution of k-means with k=13
set.seed(1002476) #Set the seed for reproducibility

# Create 10 clusters
wines_k13_centers <-kmeans(wines, centers=13) 
wines_k13_centers$centers # Display cluster centers
table(wines_k13_centers$cluster) # Give a count of data points in each cluster

# Remove columns 1 and 13
wines_k13 <-kmeans(wines[,-c(1,13)], centers=13) 
wines_k13$centers # Display cluster centers
table(wines_k13$cluster) # Give a count of data points in each cluster
table(wines_k13_centers$cluster)

# Run the algorithm for different values of k 
bss <- numeric()
wss <- numeric()

set.seed(1002476)
for(i in 1:13){
  
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

## RESULTS

# Execution of k-means with k=5
set.seed(1002476)
wines_k5 <- kmeans(wines, centers=5)

# Mean values of each cluster
aggregate(wines, by=list(wines_k5$cluster), mean)

# Clustering 
ggpairs(cbind(wines, Cluster=as.factor(wines$quality)),
        columns=1:13, aes(colour=Cluster, alpha=0.5),
        lower=list(continuous="points"),
        upper=list(continuous="blank"),
        axisLabels="none", switch="both")

