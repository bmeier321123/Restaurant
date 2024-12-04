library(vroom)
library(modeltime)
library(timetk)
library(prophet)
library(tidyverse)
library(dplyr)
library(tidymodels)
library(bonsai)
library(ggplot2)
library(lightgbm)
library(parsnip)
library(randomForest)

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

# Function to export predictions
predict_export <- function(workFlow, fileName) { 
  predictions <- predict(workFlow, new_data = testData) %>%  # Use workFlow here instead of final_wf
    bind_cols(testData) %>%
    select(Id, .pred) %>%
    rename(Prediction = .pred)  # Adjust if needed
  vroom_write(predictions, file = fileName, delim = ",")
}

# Prepare data
colnames(trainData)[2] <- "date"
colnames(testData)[2] <- "date"
trainData$date <- as.Date(trainData$date, "%m/%d/%y")
testData$date <- as.Date(testData$date, "%m/%d/%y")
colnames(trainData)[4] <- "Group"
colnames(testData)[4] <- "Group"

# Recipe
my_recipe <- recipe(revenue ~ ., data = trainData) %>%
  step_mutate_at(Id, fn = factor) %>%
  step_mutate_at(City, fn = factor) %>%
  step_mutate_at(Group, fn = factor) %>%
  step_mutate_at(Type, fn = factor) %>%
  step_date(date, features = c("dow", "month", "year", "decimal")) %>%
  step_rm(date) %>% 
  step_mutate_at(date_dow, fn = factor) %>%
  step_mutate_at(date_month, fn = factor) %>%
  step_mutate_at(date_year, fn = factor) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Define random forest model
boost_mod <- boost_tree(
  mtry = tune(),
  min_n = tune(),
  trees = 500) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_mod)

folds <- vfold_cv(trainData, v = 10, repeats = 2)

grid_of_tuning_params_b <- grid_regular(
  parameters(mtry(range = c(1, ncol(trainData) - 1)), min_n()),
  levels = 5
)


# Tuning the model
CV_results <- boost_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(rmse, mae, rsq)
  )

# Finalize the workflow with the best hyperparameters
bestTune <- CV_results %>% select_best(metric = "mae")

final_wf <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)

# Predict and export
predict_export(final_wf, "Boost_10.csv")
