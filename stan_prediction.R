# 0. Import modules
library(caret)
library(kernlab)
library(ggplot2)
library(rstan)

setwd("/Users/vaiyr630/Cood/nhl-ennustus")
# 1. Load, plot and preprocess data

## Load CSV from github
nhl202021 <- read.csv('https://raw.githubusercontent.com/ninpnin/nhl-ennustus/main/data/202021.csv')
nhl202122 <- read.csv('https://raw.githubusercontent.com/ninpnin/nhl-ennustus/main/data/202122.csv')

## Change colname +/- for convenience
colnames(nhl202122)[10] ="PLUSMINUS"
colnames(nhl202021)[10] ="PLUSMINUS"

## Drop the alphabetical player rank
nhl202122 = subset(nhl202122, select = -c(1) )
nhl202021 = subset(nhl202021, select = -c(1) )

## Plot some of the dimensions
## Plot PTS, 202122
ggplot(nhl202122, aes(x=PTS)) + 
  geom_histogram(aes(y=..density..),
                 binwidth=2.5,
                 colour="grey", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")


# Get rid of nonnumeric columns
nhl202021_num <- nhl202021[, unlist(lapply(nhl202021, is.numeric), use.names = FALSE)]
nhl202021_num

scale_data <- as.data.frame(scale(nhl202021_num))
scale_data
var(scale_data[,"Age"])
nhl_corr <- cor(nhl202021_num )
nhl_corr

## Normalize input space X to zero mean and unit variance
nhl202021_normalized <- as.data.frame(scale(nhl202021_num))
nhl202021_normalized

## Verify normalization, check two columns
mean(nhl202021_normalized[, "Age"])
var(nhl202021_normalized[, "Age"])
mean(nhl202021_normalized[, "PP"])
var(nhl202021_normalized[, "PP"])

## Replace N/A values with zeros
nhl202021_normalized[is.na(nhl202021_normalized)] <- 0

## Merge the two tables into a dataset
target_variable <- "PTS" # PTS, FO.
nhl202021_normalized["Player"] <- nhl202021["Player"]
nhl202021_normalized["PlayerID"] <- nhl202021["PlayerID"]
target_df <- nhl202122[c("Player", "PlayerID", target_variable)]
colnames(target_df)[3] ="y"
nhl_dataset <- merge(nhl202021_normalized, target_df, by = "PlayerID")
nhl_dataset <- nhl_dataset[, unlist(lapply(nhl_dataset, is.numeric), use.names = FALSE)]
nhl_dataset <- na.omit(nhl_dataset)
nhl_dataset

## Split into train and test sets
training.samples <- caret::createDataPartition(nhl_dataset$y, p=0.7, list=FALSE)
train.data <- nhl_dataset[training.samples, ]
#var(nhl202021["PTS"])
test.data <- nhl_dataset[-training.samples, ]

covariates <- c("PTS", "G", "GP", "GW", "A", "HIT", "PLUSMINUS", "S", "BLK")
train_x <- as.matrix(train.data[, covariates])
train_y <- as.matrix(train.data[,c("y")])
dims <- dim(train_x[,])
test_x <- as.matrix(test.data[, covariates])
test_y <- as.matrix(test.data[,c("y")])
dims_test <- dim(test_x[,])

train_dat <- list(x=train_x, y=c(train_y), N=dims[1], M=dims[2], x_test=test_x, N_test=dims_test[1])
fit1 <- stan(file = "model.stan", data=train_dat)
fit1
e2 <- extract(fit1, permuted = TRUE)
y_test_hat <- e2$y_test_hat
dim(y_test_hat)

y_test_hat_mean <- colMeans(y_test_hat)
length(y_test_hat_mean)
head(y_test_hat_mean,10)
sqrt(var(test_y-y_test_hat_mean))
var(test_y)
head(test_y, 10)
test_x[2,]
