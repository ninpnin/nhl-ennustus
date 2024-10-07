# 0. Import modules
# Neural networks with keras/tensorflow
library(keras)
library(tensorflow)
library(caret)
library(kernlab)
library(ggplot2)

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

# Get rid of nonnumeric columns
nhl202021_num <- nhl202021[, unlist(lapply(nhl202021, is.numeric), use.names = FALSE)]
nhl202021_num

scale_data <- as.data.frame(scale(nhl202021_num))
scale_data
var(scale_data[,"Age"])

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
dim(nhl_dataset)

## Split into train and test sets
training.samples <- caret::createDataPartition(nhl_dataset$y, p=0.7, list=FALSE)
train.data <- nhl_dataset[training.samples, ]
x_train <- data.matrix(train.data[,-24])
y_train <- c(train.data[,24])
input_shape <- dim(x_train)[2]

#var(nhl202021["PTS"])
test.data <- nhl_dataset[-training.samples, ]
data.matrix(x_train)
dim(train.data)
# Define the loss function 
loss_fn <- loss_mean_squared_error()

model <- keras_model_sequential(input_shape = input_shape) %>%
  layer_flatten() %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(10)

model %>% compile(
  optimizer = "adam",
  loss = loss_fn
)

# Overfit deep model
model %>% fit(
  x_train,
  y_train,
  epochs = 100,
  batch_size = 64#,
  #validation_data = list(x_val, y_val)
)

# Evaluate the model on the test set
model %>% evaluate(x_test,  y_test, verbose = 2)
