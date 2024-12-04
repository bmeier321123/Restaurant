library(vroom)
library(modeltime)
library(timetk)
library(prophet)
library(tidyverse)
library(dplyr)
library(tidymodels)
library(dbarts)
library(ggplot2)
library(lightgbm)
library(parsnip)
library(randomForest)

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

# Function to export predictions
predict_export <- function(workFlow, fileName) {  
  predictions <- predict(workFlow, new_data = testData) %>%  
    bind_cols(testData) %>%
    select(Id, .pred) %>%
    rename(Prediction = .pred)  
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
  step_mutate(Id = factor(Id),
              City = factor(City),
              Group = factor(Group),
              Type = factor(Type)) %>%
  step_date(date, features = c("dow", "month", "year", "decimal")) %>%
  step_rm(date) %>% 
  step_mutate(date_dow = factor(date_dow),
              date_month = factor(date_month),
              date_year = factor(date_year)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

glimpse(bake(prep(my_recipe), trainData))

# Model
preg_model <- linear_reg(penalty = 1, mixture = 0) %>% 
  set_engine("glmnet") 

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=trainData)

predict_export(preg_wf, "Pen_Reg4.csv")

