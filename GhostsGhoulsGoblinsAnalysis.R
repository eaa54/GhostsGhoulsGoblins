library(vroom)
library(embed)
library(tidymodels)
library(tidyverse)
library(embed)
library(discrim)

train <- vroom("C:/Users/eaa54/Documents/School/STAT348/GhostsGhoulsGoblins/train.csv")
test <- vroom("C:/Users/eaa54/Documents/School/STAT348/GhostsGhoulsGoblins/test.csv")
#missing <- vroom("C:/Users/eaa54/Documents/School/STAT348/GhostsGhoulsGoblins/trainWithMissingValues.csv")

# Create recipe
ggg_recipe <- recipe(type ~ ., train) %>%
  step_mutate(type = as.factor(type), skip = TRUE) %>%
  step_mutate(bonehair = bone_length*hair_length) %>%
  step_mutate(color = as.factor(color)) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type))

baked <- bake(prep(ggg_recipe), train)

##Class Competition Model
# Use Linear SVM model
svmLinear <- svm_linear(cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

# set workflow
svm_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(svmLinear)

# tune cost
# Set up tuning values
svm_grid <- grid_regular(cost(), levels = 3)

# Set up k-fold cross validation and run it
svm_folds <- vfold_cv(train, v = 2, repeats = 1)

CV_svm_results <- svm_wf %>%
  tune_grid(resamples = svm_folds,
  grid = svm_grid,
  metrics = metric_set(accuracy)) #accuracy is a measure of correct predictions

# Find Best Tuning Parameters
bestTune_svm <- CV_svm_results %>%
select_best("accuracy")

#finalize workflow and fit it
final_svm_wf <- svm_wf %>%
  finalize_workflow(bestTune_svm) %>%
  fit(train)

pred_svm <- predict(final_svm_wf, new_data = test, type = "class") %>%
  bind_cols(., test) %>%
  rename(type = .pred_class) %>%
  select(id, type)

vroom_write(pred_svm, "GGG_preds_svm.csv", delim = ",")

###################
##NEURAL NETWORKS##
###################
nn_recipe <- recipe(type ~ ., train) %>%
  update_role(id, new_role="id") %>%
  step_mutate(type = as.factor(type), skip = TRUE) %>%
  step_mutate(color = as.factor(color)) %>%
  step_range(all_numeric_predictors(), min=0, max=1) %>% #scale to [0,1]
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type))

#try taking out color
#try nb including id
#maybe mess with number of folds (maybe less)
# Neural Network Model
nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>% #or 100 or 250
  set_engine("nnet") %>% 
  set_mode("classification")

# set workflow
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 75)),
                            levels=3)

# Set up k-fold cross validation and run it
nn_folds <- vfold_cv(train, v = 5, repeats = 1)

CV_nn_results <- nn_wf %>%
  tune_grid(resamples = nn_folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

CV_nn_results %>% collect_metrics() %>% filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

# Find Best Tuning Parameters
bestTune_nn <- CV_nn_results %>%
  select_best("accuracy")

#finalize workflow and fit it
final_nn_wf <- nn_wf %>%
  finalize_workflow(bestTune_nn) %>%
  fit(train)

pred_nn <- predict(final_nn_wf, new_data = test, type = "class") %>%
  bind_cols(., test) %>%
  rename(type = .pred_class) %>%
  select(id, type)

vroom_write(pred_nn, "GGG_preds_nn.csv", delim = ",")

############
##BOOSTING##
############
library(bonsai)
library(lightgbm)

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
              set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
              set_mode("classification")

## CV tune, finalize and predict here and save result
# set workflow
boost_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(boost_model)

boost_tuneGrid <- grid_regular(tree_depth(), trees(), learn_rate(), levels=3)

# Set up k-fold cross validation and run it
boost_folds <- vfold_cv(train, v = 5, repeats = 1)

CV_boost_results <- boost_wf %>%
  tune_grid(resamples = boost_folds,
            grid = boost_tuneGrid,
            metrics = metric_set(accuracy))

# Find Best Tuning Parameters
bestTune_boost <- CV_boost_results %>%
  select_best("accuracy")

#finalize workflow and fit it
final_boost_wf <- boost_wf %>%
  finalize_workflow(bestTune_boost) %>%
  fit(train)

pred_boost <- predict(final_boost_wf, new_data = test, type = "class") %>%
  bind_cols(., test) %>%
  rename(type = .pred_class) %>%
  select(id, type)

vroom_write(pred_boost, "GGG_preds_boost.csv", delim = ",")

########
##BART##
########
library(dbarts)

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

## CV tune, finalize and predict here and save result
# set workflow
bart_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(bart_model)

bart_tuneGrid <- grid_regular(trees(), levels=3)

# Set up k-fold cross validation and run it
bart_folds <- vfold_cv(train, v = 5, repeats = 1)

CV_bart_results <- bart_wf %>%
  tune_grid(resamples = bart_folds,
            grid = bart_tuneGrid,
            metrics = metric_set(accuracy))

# Find Best Tuning Parameters
bestTune_bart <- CV_bart_results %>%
  select_best("accuracy")

#finalize workflow and fit it
final_bart_wf <- bart_wf %>%
  finalize_workflow(bestTune_bart) %>%
  fit(train)

pred_bart <- predict(final_bart_wf, new_data = test, type = "class") %>%
  bind_cols(., test) %>%
  rename(type = .pred_class) %>%
  select(id, type)

vroom_write(pred_bart, "GGG_preds_bart.csv", delim = ",")

######
##NB##
######
# model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") #need discrim library for this engine

# set workflow
nb_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(nb_model)

# tune smoothness and Laplace here
# Set up tuning values
nb_grid <- grid_regular(Laplace(),
                        smoothness(),
                        levels = 3)

# Set up k-fold cross validation and run it
nb_folds <- vfold_cv(train, v = 3, repeats = 1)

CV_nb_results <- nb_wf %>%
  tune_grid(resamples = nb_folds,
            grid = nb_grid,
            metrics = metric_set(roc_auc)) #area under ROC curve (false positives vs. true positives)

# Find Best Tuning Parameters
bestTune_nb <- CV_nb_results %>%
  select_best("roc_auc")

# finalize workflow and fit it
final_nb_wf <- nb_wf %>%
  finalize_workflow(bestTune_nb) %>%
  fit(data = train)

pred_nb <- predict(final_nb_wf, new_data = test, type = "class") %>%
  bind_cols(., test) %>%
  rename(type = .pred_class) %>%
  select(id, type)

vroom_write(pred_nb, "GGG_nb_preds.csv", delim = ",")




