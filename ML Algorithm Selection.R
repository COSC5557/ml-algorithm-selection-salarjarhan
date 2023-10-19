# Load libraries
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3misc)
library(mlr3viz)
library(readr)
library(caret)
library(ggplot2)
library(dplyr)
library(gridExtra)


# Load the wine quality data from a CSV file
wine_data <- read.csv("winequality-white.csv", sep= ";")  

# For reproducibility
set.seed(123)

# Split the data into a training set and a testing set
train_index <- createDataPartition(wine_data$quality, p = 0.8, list = FALSE)
train_data <- wine_data[train_index, ]
test_data <- wine_data[-train_index, ]

# Create a regression Task object
task = as_task_regr(train_data, target = "quality")
test_task <- as_task_regr(test_data, target = "quality")
# Define regression learners with specified hyperparameters
models <- list(
  lrn("regr.kknn", k = 5, distance = 2),  # k-Nearest Neighbors with k=5
  lrn("regr.lm"),           # Linear regression (no specific hyperparameters)
  lrn("regr.ranger", num.trees = 100, mtry = 4),  # Random Forest with 100 trees and mtry=4
  lrn("regr.svm", type = "eps-regression", kernel = "linear", cost = 1.0, epsilon =	0.1),  # Support Vector Machine with linear kernel and cost=1.0
  lrn("regr.xgboost", nrounds = 100, max_depth = 6)  # Gradient Boosting with 100 rounds and max_depth=6
)

# Define resampling strategy (e.g., 5-fold cross-validation)
resampling_strategy <- rsmp("cv", folds = 5)

# Train and evaluate models
results <- lapply(models, function(learner) {
  # Train the model
  learner$train(task)
  
  # Perform the resampling and evaluation
  resampling <- resample(task, learner, resampling_strategy, store_models = TRUE)
  
  # Extract performance metrics (RMSE in this case)
  performance <- resampling$aggregate(msr("regr.rmse"))
  
  # Predict on the test data
  predictions <- predict(learner, test_task$data())
  rmse_test <- sqrt(mean((test_data$quality - predictions)^2))
  
  # Return evaluation results
  list(
    learner = learner,
    resampling = resampling,
    performance = performance,
    rmse_test = rmse_test
  )
})

# Display evaluation results
for (result in results) {
  cat("Learner:", result$learner$id, "\n")
  cat("RMSE for 5-fold CV:", result$performance, "\n")
  cat("RMSE for validation:", result$rmse_test, "\n")
  cat("\n")
}

# Create a list of plots 
plots <- list()
for (i in 1:length(results)) {
  # Generate the plot and store it in the plots
  plots[[i]] <- autoplot(results[[i]]$learner$train(task)$predict(task), type = "xy") + ggtitle(results[[i]]$learner$id)
}
grid.arrange(grobs = plots, ncol = 3, nrow = 2)

#References
#https://mlr3learners.mlr-org.com/
#https://mlr3.mlr-org.com/reference/predict.Learner.html
#https://mlr-org.com/gallery/technical/2022-12-22-mlr3viz
#https://chat.openai.com/share/c8ff1026-84fc-4ca6-ab47-88017e8c25ff

