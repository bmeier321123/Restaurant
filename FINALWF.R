library(vroom)
library(parsnip)
library(recipes)
library(dplyr)
library(tidymodels)
library(randomForest)
library(lightgbm)
library(glmnet)

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

predict_export <- function(workFlow, fileName) {
  predictions <- predict(final_wf, new_data = testData) %>%
    bind_cols(testData) %>%
    select(Id, .pred) %>%  # Adjust according to your actual prediction columns
    rename(Prediction = .pred)  # Adjust if needed
  vroom_write(predictions, file = fileName, delim = ",")
}

colnames(trainData)[2] = "date"
colnames(testData)[2] = "date"

trainData$date <- as.Date(trainData$date, "%m/%d/%y")
testData$date <- as.Date(testData$date, "%m/%d/%y")

colnames(trainData)[4] = "Group"
colnames(testData)[4] = "Group"

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


set.seed(1999)

ran_for_mod <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000) %>%
  set_engine("randomForest") %>%
  set_mode("regression")

ran_for_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(ran_for_mod)

grid_of_tuning_params <- grid_regular(
  parameters(mtry(range = c(1, ncol(trainData) - 1)), min_n()),
  levels = 5
)

folds <- vfold_cv(trainData, v = 5, repeats = 2)

CV_results <- ran_for_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(rmse, mae, rsq)
  )

## Finalize workflow and predict
bestTune <- CV_results %>%
  select_best(metric = "rsq")

## Set up grid of tuning values
final_wf <-ran_for_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

## Finalize workflow and predict
final_wf %>%
  predict(new_data = testData)

predict_export(final_wf, "RandomForest_Final.csv")
