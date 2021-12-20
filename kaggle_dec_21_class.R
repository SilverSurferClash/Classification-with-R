
library(tidyverse)
library(tidymodels)


# Read in a CSV file

df_raw <- read_csv("data/train.csv")


# Covert to a dataframe or tibble
# Facorize target variable

df <- as_tibble(df_raw) 
df$Cover_Type<- as_factor(df$Cover_Type)
df <- df %>% rename(target = Cover_Type ) %>% select(target, everything())

## S3 method for workflow
#tune_grid(
 # object,      # workflow object
  #resamples,   # rsamples object
  #...,
  #param_info = NULL,  # dials:parameters object
  # grid = 10, # dataframe of tuning values
  # metrics = NULL,  # yardstick:metric_set()
  # control = control_grid() # object for modifying the tuning process
#)


# Build the Workflow object

# 1) Build  the first  model - Randon Forest

rf_mod <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# 2) Build the recipe

recipe_rf <- recipe(target ~ ., data = train_df) %>% step_nzv(all_predictors()) %>% step_normalize(all_predictors()) %>% step_corr(all_predictors())

# 3) Build the workflow

wf_rf <- workflow() %>% add_recipe(recipe_rf) %>% add_model(rf_mod)

## Build the resamples object

set.seed(23)

split_df <- df %>% initial_split()

train_df <- training(split_df)

test_df <- testing(split_df)

set.seed(21)

fold_df <- vfold_cv(train_df, v= 5)

# Define the grid object

rf_params <- 
  dials::parameters(
    mtry = tune(),
    min_n = tune()
  )

rf_grid <- 
  dials::grid_max_entropy(
    rf_params, 
    size = 20 # check the best value here (xgboost article used 60)
  )

# Define the control grid

ctrl_features <- control_grid(verbose = TRUE)


# Define the tune_grid object

tune_results <- tune_grid(
  object = wf_rf,
  resamples = fold_df,
  grid = rf_grid,
  control = ctrl_features
  
)


#tune_grid(
# object,      # workflow object
#resamples,   # rsamples object
#...,
#param_info = NULL,  # dials:parameters object
# grid = 10, # dataframe of tuning values
# metrics = NULL,  # yardstick:metric_set()
# control = control_grid() # object for modifying the tuning process
#)






