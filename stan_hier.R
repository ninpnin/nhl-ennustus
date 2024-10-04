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
colnames(nhl202122)
colnames(nhl202122)[10] ="PLUSMINUS"
colnames(nhl202021)[10] ="PLUSMINUS"

## Drop the alphabetical player rank
nhl202122 = subset(nhl202122, select = -c(1) )
nhl202021 = subset(nhl202021, select = -c(1) )

target_variable <- "PTS" # PTS, FO.
target_df <- nhl202122[c("Player", "PlayerID", target_variable)]
colnames(target_df)[3] ="y"
nhl202021_alt <- merge(nhl202021, target_df, by = "PlayerID")

# Get rid of nonnumeric columns
position_map <- data.frame(Pos=c("C", "RW", "LW", "D", "F", "W"), PosNum= c(1, 2, 3, 4, 5, 6))
nhl202021_alt <- merge(nhl202021_alt, position_map, by="Pos", all.x = TRUE)
nhl202021_num <- nhl202021_alt[, unlist(lapply(nhl202021_alt, is.numeric), use.names = FALSE)]
head(nhl202021_num)

## Normalize input space X to zero mean and unit variance
covariates <- c("PTS", "G", "GP", "GW", "A", "HIT", "PLUSMINUS", "S", "BLK", "PP", "TOI", "EV", "FO.")
#covariates <- c("PTS", "G", "GP", "GW", "A", "HIT", "PLUSMINUS", "S", "BLK", "PP", "TOI", "EV", "FO.")
#covariates <- c("PTS", "PP")
nhl202021_normalized <- as.data.frame(nhl202021_num)
names(nhl202021_normalized)
nhl202021_normalized[covariates] <- scale(nhl202021_normalized[covariates])
head(nhl202021_normalized)

## Verify normalization, check two columns
mean(nhl202021_normalized[, "PP"])
var(nhl202021_normalized[, "PP"])

## Replace N/A values with zeros
nhl202021_normalized[is.na(nhl202021_normalized)] <- 0

## Merge the two tables into a dataset

## Split into train and test sets
training.samples <- caret::createDataPartition(nhl_dataset$y, p=0.7, list=FALSE)
train.data <- nhl_dataset[training.samples, ]
#var(nhl202021["PTS"])
test.data <- nhl_dataset[-training.samples, ]

train_x <- as.matrix(train.data[, covariates])
train_z <- as.matrix(train.data[, c("PosNum")])
train_y <- as.matrix(train.data[,c("y")])
dims <- dim(train_x[,])
test_x <- as.matrix(test.data[, covariates])
test_z <- as.matrix(test.data[, c("PosNum")])
test_y <- as.matrix(test.data[,c("y")])
dims_test <- dim(test_x[,])

cor(train.data[c("PTS", "y")])

train_dat <- list(x=train_x, y=c(train_y), z=c(train_z) , N=dims[1], M=dims[2], x_test=test_x,z_test=c(test_z), N_test=dims_test[1])
fit1 <- stan(file = "hierarchical_model.stan", data=train_dat, iter=4000)
fit1
e2 <- extract(fit1, permuted = TRUE)
y_test_hat <- e2$y_test_hat
dim(y_test_hat)

y_test_hat_mean <- colMeans(y_test_hat)
length(y_test_hat_mean)
head(y_test_hat_mean,10)
sqrt(var(test_y-y_test_hat_mean))
sqrt(var(test_y))
head(test_y, 10)
test_x[2,]
