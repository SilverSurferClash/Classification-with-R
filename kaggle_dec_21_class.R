
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
  set_engine("randomForest")



## Build the resamples object

set.seed(23)

df <- df %>% sample_frac(0.1)

split_df <- df %>% initial_split()

train_df <- training(split_df)

test_df <- testing(split_df)

set.seed(21)

fold_df <- vfold_cv(train_df, v= 5)

# 2) Build the recipe

recipe_rf <- recipe(target ~ ., data = train_df) %>% step_nzv(all_predictors()) %>% step_normalize(all_predictors()) %>% step_corr(all_predictors())

# 3) Build the workflow

wf_rf <- workflow() %>% add_recipe(recipe_rf) %>% add_model(rf_mod)

# Define the grid object

rf_params <- parameters(finalize(mtry(), select(df, -target)) , min_n())



rf_grid <- grid_max_entropy(rf_params, size = 20)

# Define the control grid

ctrl_features <- control_grid(verbose = TRUE, save_pred = TRUE)


# Initialize the parallel computing
library(doParallel)

all_cores <- parallel::detectCores(logical = FALSE)

cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

clusterEvalQ(cl, {library(tidymodels)})


# Define the tune_grid object

start_time <- Sys.time()

tune_results <- tune_grid(
  object = wf_rf,
  resamples = fold_df,
  grid = rf_grid,
  control = ctrl_features
)

end_time <- Sys.time()

print(end_time - start_time)




best_params <- tune_results %>% select_best("accuracy")


final_wf <- wf_rf %>% finalize_workflow(best_params)


final_fit_rf <- final_wf %>% last_fit(split_df)


final_fit_rf %>%
  collect_metrics()


final_fit_rf %>%
  collect_predictions() %>% 
  roc_curve(class, .pred_PS) %>% 
  autoplot()





