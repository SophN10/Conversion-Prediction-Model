# Load necessary packages
library(caret)
library(gt)
library(pROC)
library(glmnet)
library(MASS)
library(dplyr)
library(stringr)
library(ggplot2)
library(PRROC)
library(scales)
library(broom)
library(tuneRanger)
library(tibble)
library(xgboost)
library(Matrix)
library(patchwork)
library(SHAPforxgboost)
library(data.table)

# Load balanced training data 
training_data <- readRDS("C:\\Users\\sophi\\OneDrive - University of Surrey
                         \\Documents\\Machine Learning Coursework Dataset\\
                         balanced_data.rds")

#===============================================================================
# Train full logistic regression model with 10-fold CV 
#===============================================================================

# Set seed for reproducibility
set.seed(42)
training_data$Is_Revenue <- factor(training_data$Is_Revenue, 
                                   levels = c("No", "Yes"))

train_control <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, 
  savePredictions = TRUE
)

# Train full glm model
logit_model <- train(
  Is_Revenue ~ .,
  data = training_data,
  method = "glm",
  family = "binomial",
  trControl = train_control,
  metric = "ROC"
)

print(logit_model)
summary(logit_model)

#===============================================================================
# Train glmnet (Lasso) model with 10-fold CV 
#===============================================================================

# Set seed for reproducibility
set.seed(42)

train_control <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, 
  verboseIter = TRUE, 
  savePredictions = TRUE
)


tune_grid <- expand.grid(
  alpha = 1, # Lasso regression
  lambda = seq(0.0001, 0.1, length.out = 10) 
)


x <- model.matrix(Is_Revenue ~ ., data = training_data)[, -1] 
y <- factor(training_data$Is_Revenue, levels = c("No", "Yes")) 


lasso_model <- train(
  x = x,
  y = y,
  method = "glmnet",
  family = "binomial",
  trControl = train_control,
  tuneGrid = tune_grid, 
  metric = "ROC"
)

print(lasso_model$bestTune)
print(coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda))

#===============================================================================
# Build a standard logistic regression with the variables selected by lasso
#===============================================================================

coefs <- coef(lasso_model$finalModel, s = 0.0001)
coefs_vec <- as.vector(coefs)
var_names <- rownames(coefs)

selected_vars <- var_names[which(coefs_vec !=0)]
selected_vars <- selected_vars[selected_vars != "(Intercept)"]

x_selected <- x[, selected_vars, drop = FALSE]

glm_data <- data.frame(Is_Revenue = y, x_selected)

glm_model <- glm(Is_Revenue ~ ., data = glm_data, family = binomial)

summary(glm_model)

# Stepwise AIC for model refinement 
glm_step <- stepAIC(glm_model, direction = "both")
summary(glm_step)

#===============================================================================
# Function to automatically remove insignificant variables based on P-value 
#===============================================================================

auto_remove_insignificant <- function(glm_data, response = "Is_Revenue", 
                                      p_threshold = 0.05, max_iter = 10){
  formula <- as.formula(paste(response, "~."))
  current_data <- glm_data
  iter <- 0
  
  repeat {
    iter <- iter + 1
    glm_model <- glm(formula, data = current_data, family = binomial)
    summary_model <- summary(glm_model)
    
    # Extract coefficients table excluding intercept
    coefs <- summary_model$coefficients[-1, , drop = FALSE]
    
    # Stop if all p-values are below threshold or max iterations reached
    if (all (coefs[, "Pr(>|z|)"] <= p_threshold) || iter > max_iter){
      break
    }
    
    # Variable with the highest p-value above threshold 
    max_p_var <- rownames(coefs)[which.max(coefs[, "Pr(>|z|)"])]
    
    message(sprintf("Removing variable '%s' with p-value %.4f",
                    max_p_var, coefs[max_p_var, "Pr(>|z|)"]))
    
    # Remove that variable from the data frame
    current_data <- current_data[, !(names(current_data) %in% max_p_var), 
                                 drop = FALSE]
  }
  
  return(list(final_model = glm_model, final_data = current_data))
}

# Run automatic variable removal function
result <- auto_remove_insignificant(glm_data, "Is_Revenue", p_threshold = 0.05)
print(head(result$final_data))
print(summary(result$final_model))

#===============================================================================
# Extract and prepare coefficient for feature importance plot  
#===============================================================================

coef_df <- broom::tidy(result$final_model) %>% 
  filter(term !="(Intercept)") %>%
  mutate(
    Feature = str_replace_all(term, "_", " "), 
    Importance = abs(estimate) 
  )%>%
  arrange(desc(Importance)) %>%
  head(10)%>% # top 10
  mutate(Feature = factor(Feature, levels = rev(Feature))) %>% 
  mutate(Importance = scales::rescale(Importance, to = c(0,1)))

# Plot Feature Importance
ggplot(coef_df, aes(x = Feature, y = Importance, fill = Importance))+
  geom_col()+
  coord_flip()+
  labs(title = "Feature Importance from Logistic Regression Model", 
       x = "Features",
       y = "Absolute Coefficient")+
  scale_fill_gradient(low = "lightblue", high = "darkblue")+
  theme_classic()+
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# Compute predicted probabilities and ROC for final glm model  
#===============================================================================

predicted_probs <- predict(result$final_model, type = "response")

observed_labels <- factor(result$final_data$Is_Revenue, levels = c("No", "Yes"))

# Compute ROC object
roc_obj <- roc(observed_labels, predicted_probs, direction = "<", 
               levels = c("No", "Yes"))

roc_points <- coords(roc_obj, "all", ret = c("specificity", "sensitivity"))

roc_df_glm <- data.frame(
  specificity = roc_points$specificity, 
  sensitivity = roc_points$sensitivity, 
  Model = "Logistic Regression"
)

# Plot ROC curve for logistic regression model
ggplot(roc_df_glm, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue", lwd = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "ROC Curve for Logistic Regression Model",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_classic() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# Extract predictions from final model and plot ROC objects per Fold   
#===============================================================================

lasso_preds <- lasso_model$pred

lasso_preds$obs <- factor(lasso_preds$obs, levels = c("No", "Yes"))
lasso_preds$Yes <- as.numeric(lasso_preds$Yes) 

roc_list <- lasso_preds %>%
  group_by(Resample) %>%
  summarise(roc_obj = list(roc(obs, Yes)))

roc_points <- do.call(rbind, lapply(1:nrow(roc_list), function(i) {
  roc_obj <- roc_list$roc_obj[[i]]
  coords_df <- coords(roc_obj, "all", ret = c("specificity", "sensitivity"))
  data.frame(
    specificity = coords_df$specificity, 
    sensitivity = coords_df$sensitivity, 
    fold = roc_list$Resample[i]
  )
}))

# Plot ROC Curve per fold 
ggplot(roc_points, aes(x = 1 - specificity, y= sensitivity, color = fold))+
  geom_line(linewidth = 1)+
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray")+
  labs(title = "ROC Curve per Fold for Logistic Regression Model", 
       x = "False Positive Rate", 
       y = "True Positive Rate")+
  theme_classic ()+
  theme(legend.position = "bottom") +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# Build Precision-Recall (PR Curve)
#===============================================================================

predicted_probs <- predict(result$final_model, newdata = result$final_data, 
                           type = "response")

actuals <- ifelse(result$final_data$Is_Revenue == "Yes", 1, 0)
table(actuals)
 
pr <- pr.curve(scores.class0 = predicted_probs[actuals == 1], 
               scores.class1 = predicted_probs[actuals == 0], 
               curve = TRUE)

pr_df_glm <- data.frame(Recall = pr$curve[, 1], 
                        Precision = pr$curve[, 2], 
                        Model = "Logistic Regression")

# Plot PR Curve
ggplot(pr_df_glm, aes(x = Recall, y = Precision))+
  labs(title = "Precision-Recall Curve", x = "Recall", y = "Precision")+
  geom_line(color = "blue", lwd = 1)+
  theme_minimal()
#===============================================================================
# Generate Predictions and Confusion Matrix 
#===============================================================================

pred_classes <- ifelse(predicted_probs > 0.5, "Yes", "No")

pred_factor <- factor(pred_classes, levels = c("No", "Yes"))
observed_labels <- factor(result$final_data$Is_Revenue, levels = c("No", "Yes"))

# Create a confusion matrix 
conf_matrix <- confusionMatrix(data = pred_factor, 
                               reference = observed_labels, 
                               positive = "Yes")

print(conf_matrix)

cm_df <- as.data.frame(conf_matrix$table)

colnames(cm_df) <- c("Predicted", "Actual", "Count")

# Create a nice-looking table for the confusion matrix 
cm_df %>%
  gt() %>%
  tab_header(
    title = "Confusion Matrix"
  ) %>%
  cols_label(
    Predicted = "Predicted Class",
    Actual = "Actual Class",
    Count = "Number of Observations"
  ) %>%
  fmt_number(columns = c(Count), decimals = 0) %>%
  data_color(
    columns = c(Count),
    fn = scales::col_numeric(
      palette = c("white", "steelblue"),
      domain = NULL
    )
  )

#Performance Metrics Calculations
precision <- as.numeric(conf_matrix$byClass["Precision"])
recall <- as.numeric(conf_matrix$byClass["Sensitivity"])
F1 <- 2 * (precision * recall)/ (precision + recall)

roc_auc <- auc(roc_obj)

pr_auc <- pr$auc.integral

metrics_glm <- data.frame(
  Accuracy = conf_matrix$overall["Accuracy"],
  Recall = conf_matrix$byClass["Sensitivity"],
  Precision = conf_matrix$byClass["Precision"],
  F1 = F1, 
  ROC_AUC = roc_auc,
  PR_AUC = pr_auc
)

print(metrics_glm)

# Create a nice looking table for performance metrics
metrics_glm %>%
  gt() %>%
  tab_header(
    title = "Logistic Regression Model Performance Metrics"
  ) %>%
  fmt_number(columns = everything(), decimals = 3) %>%
  cols_label(
    Accuracy = "Accuracy",
    Recall = "Recall",
    Precision = "Precision",
    F1 = "F1 Score", 
    ROC_AUC = "ROC AUC", 
    PR_AUC = "PR AUC"
  ) %>%
  data_color(
    columns = everything(),
    fn = scales::col_numeric(
      palette = c("#90caf9"),
      domain = c(0,1)
    )
  )

#===============================================================================
# Train Random Forest Model with Best Parameters
#===============================================================================

# Set seed for reproducibility
set.seed(42)

train_control <- trainControl(method = "cv", 
                              number = 10, 
                              classProbs = TRUE,
                              verboseIter = TRUE,
                              summaryFunction = twoClassSummary,
                              savePredictions = TRUE)

# Define the tuning grid 
tune_grid <- expand.grid(
  mtry = c(2, 4, 6), 
  splitrule = c("gini"), 
  min.node.size = c(5, 10) 
)

# Train rf model with tuning grid
rf_model <- train(
  Is_Revenue ~ .,
  data = training_data,
  method = "ranger",
  metric = "ROC",
  tuneGrid = tune_grid,
  importance = "permutation",
  trControl = train_control
)

# Check for best tuning parameters
best_params <- rf_model$bestTune
best_params

# Train the final model with best tuning parameters 
final_rf_model <- train(
  Is_Revenue ~ .,
  data = training_data,
  method = "ranger",
  metric = "ROC",
  tuneGrid = best_params,
  importance = "permutation", 
  trControl = train_control
)

final_rf_model$result
print(rf_model)

#===============================================================================
# Generate ROC Curve for Random Forest Model Performance
#===============================================================================

roc_obj <- roc(final_rf_model$pred$obs, final_rf_model$pred$Yes)

roc_points <- coords(roc_obj, "all", ret = c("specificity", "sensitivity"))

roc_df_rf <- data.frame(
  specificity = roc_points$specificity, 
  sensitivity = roc_points$sensitivity, 
  Model = "Random Forest"
)

ggplot(roc_df_rf, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue", lwd = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "ROC Curve for Random Forest Model Performance",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_classic() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# Generate ROC Curve per Fold for Random Forest Performance  
#===============================================================================
cv_preds <- final_rf_model$pred

cv_preds$obs <- factor(cv_preds$obs, levels = c("No", "Yes"))
cv_preds$Yes <- as.numeric(cv_preds$Yes)  

roc_list <- cv_preds %>%
  group_by(Resample) %>%
  summarise(roc_obj = list(roc(obs, Yes)))

# Extract ROC points for plotting
roc_points_rf <- do.call(rbind, lapply(1:nrow(roc_list), function(i) {
  roc_obj <- roc_list$roc_obj[[i]]
  coords_df <- coords(roc_obj, "all", ret = c("specificity", "sensitivity"))
  data.frame(
    specificity = coords_df$specificity, 
    sensitivity = coords_df$sensitivity, 
    fold = roc_list$Resample[i]
  )
}))

# Plot ROC Curves for each fold 
ggplot(roc_points_rf, aes(x = 1 - specificity, y= sensitivity, color = fold))+
  geom_line(linewidth = 1)+
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray")+
  labs(title = "ROC Curve per Fold for Random Forest Model", 
       x = "False Positive Rate", 
       y = "True Positive Rate")+
  theme_classic ()+
  theme(legend.position = "bottom") +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# Build Precision-Recall (PR Curve)
#===============================================================================
rf_preds <- final_rf_model$pred

rf_probs <- final_rf_model$pred$Yes
rf_actuals <- ifelse(rf_preds$obs == "Yes", 1, 0)

# Compute PR Curve
pr <- pr.curve(scores.class0 = rf_probs[rf_actuals == 1],
               scores.class1 = rf_probs[rf_actuals == 0],
               curve = TRUE)

pr_df_rf <- data.frame(Recall = pr$curve[, 1], 
                       precision = pr$curve[, 2], 
                       Model = "Random Forest")

# Plot PR Curve
ggplot(pr_df_rf, aes(x = Recall, y = precision))+
  geom_line(color = "blue", size = 1)+
  labs(title = "Precision-Recall Curve", x = "Recall", y = "Precision")+
  theme_minimal()

#===============================================================================
# Generate Predictions and Confusion Matrix 
#===============================================================================

final_rf_model$pred$obs <- factor(final_rf_model$pred$obs, 
                                  levels = c("No", "Yes"))
final_rf_model$pred$pred <- factor(final_rf_model$pred$pred, 
                                   levels = c("No", "Yes"))

# Create confusion matrix 
conf_matrix <- confusionMatrix(data = final_rf_model$pred$pred, 
                               reference = final_rf_model$pred$obs, 
                               positive = "Yes")

colnames(cm_df) <- c("Predicted", "Actual", "Count")

# Create a nice-looking table for the confusion matrix 
cm_df %>%
  gt() %>%
  tab_header(
    title = "Confusion Matrix"
  ) %>%
  cols_label(
    Predicted = "Predicted Class",
    Actual = "Actual Class",
    Count = "Number of Observations"
  ) %>%
  fmt_number(columns = c(Count), decimals = 0) %>%
  data_color(
    columns = c(Count),
    fn = scales::col_numeric(
      palette = c("white", "steelblue"),
      domain = NULL
    )
  )

# Compute performance metrics
# Calculate F1 score
precision <- as.numeric(conf_matrix$byClass["Precision"])
recall <- as.numeric(conf_matrix$byClass["Sensitivity"])
F1 <- 2 * (precision * recall)/ (precision + recall)

# Calculate ROC AUC 
roc_obj <- roc(final_rf_model$pred$obs, final_rf_model$pred$Yes)
roc_auc <- auc(roc_obj)
print(roc_auc)

# Calculate PR AUC
pr_auc <- pr$auc.integral
print(pr_auc)

metrics_rf <- data.frame(
  Accuracy = conf_matrix$overall["Accuracy"],
  Recall = conf_matrix$byClass["Sensitivity"],
  Precision = conf_matrix$byClass["Precision"],
  F1 = F1, 
  ROC_AUC = roc_auc, 
  PR_AUC = pr_auc
)

print(metrics_rf)

# Create a nice looking table for additional metrics
metrics_rf %>%
  gt() %>%
  tab_header(
    title = "Random Forest Model Performance Metrics"
  ) %>%
  fmt_number(columns = everything(), decimals = 3) %>%
  cols_label(
    Accuracy = "Accuracy",
    Recall = "Recall",
    Precision = "Precision",
    F1 = "F1 Score", 
    ROC_AUC = "ROC AUC", 
    PR_AUC = "PR AUC"
  ) %>%
  data_color(
    columns = everything(),
    fn = scales::col_numeric(
      palette = c("#90caf9"),
      domain = c(0,1)
    )
  )

#===============================================================================
# Generate Variable Importance Plot from Final Random Forest Model 
#===============================================================================

# Create an absolute feature importance plot based on final rf model
Var_importance <- varImp(final_rf_model)

importance_df_rf <- as.data.frame(Var_importance$importance)
importance_df_rf <- importance_df_rf%>%
  rownames_to_column(var = "Feature") %>%
  arrange(desc(Overall))

# Only include the top ten features 
importance_df_rf <- importance_df_rf %>%
  head(10) %>%
  mutate(Feature = str_replace_all(Feature, "_", " ")) 

importance_df_rf$Feature <- factor(importance_df_rf$Feature, 
                                   levels = importance_df_rf$Feature)

importance_df_rf$Overall <- scales::rescale(importance_df_rf$Overall, 
                                            to = c(0, 1))

# Create a bar plot for feature importance
ggplot(importance_df_rf, aes(x = reorder(Feature, Overall), 
                             y = Overall, fill = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance from Random Forest Model",
       x = "Features",
       y = "Importance Score") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_classic() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# Train XGBoost Model with the Best Tuning Parameters 
#===============================================================================

# Set seed for reproducibility
set.seed(42)

labels <- training_data$Is_Revenue
features <- training_data[, setdiff(names(training_data), "Is_Revenue")]
labels <- as.numeric(as.factor(labels)) - 1

features_matrix <- model.matrix(~ . - 1, data = features)

dtrain <- xgb.DMatrix(data = features_matrix, label = labels)

best_auc  <- 0
best_params <- list()
best_nrounds <- 0

results <- data.frame()

xgb_grid <- expand.grid (
  nrounds = c(100), 
  max_depth = c(4, 6),
  gamma = c(0),
  eta = c(0.1), 
  colsample_bytree = c(0.8), 
  min_child_weight = c(1), 
  subsample = c(0.8)
)

for (i in 1:nrow(xgb_grid)) {
  params <- list(
    booster = "gbtree", 
    objective = "binary:logistic", 
    eval_metric = "auc", 
    eta = xgb_grid$eta[i],
    max_depth = xgb_grid$max_depth[i], 
    gamma = xgb_grid$gamma[i],
    colsample_bytree = xgb_grid$colsample_bytree[i], 
    min_child_weight = xgb_grid$min_child_weight[i], 
    subsample = xgb_grid$subsample[i]
  )
  
  cv <- xgb.cv(
    params = params, 
    data = dtrain, 
    nrounds = xgb_grid$nrounds[i], 
    nfold = 10, 
    early_stopping_rounds = 10, 
    stratified = TRUE,
    prediction = TRUE, 
    verbose = TRUE
  )
  
  best_iter <- cv$best_iteration
  best_score <- max(cv$evaluation_log$test_auc_mean)
  
  results <- rbind(results, cbind(xgb_grid[i, ], best_iter, best_score))
  
  if (best_score > best_auc) {
    best_auc <- best_score
    best_params <- params 
    best_nrounds <- best_iter
  }
}

print(paste("Best AUC:", best_auc))
print(paste("Best nrounds:", best_nrounds))
print(best_params)

# Train final model on full training data 
final_xgb_model <- xgb.train(
  params = best_params, 
  data = dtrain, 
  nrounds = 96, 
  verbose = TRUE
)

#===============================================================================
# Compute SHAP Values and Create Feature Plot for Grouped Variables
#===============================================================================

features_matrix <- model.matrix (~ . -1, data = features)

# Compute SHAP Values
shap_values <- shap.values(xgb_model = final_xgb_model, 
                           X_train = features_matrix)

shap_long <- shap.prep(shap_contrib = shap_values$shap_score, 
                       X_train = features_matrix)

shap_long <- shap_long %>%
  mutate(orig_feature = str_remove(variable, "_[^_]+$"))

# Aggregate SHAP Values by original feature 
agg_shap <- shap_long %>% 
  group_by(orig_feature) %>%
  summarise(mean_abs_shap = mean(abs(value), na.rm = TRUE)) %>%
  arrange(desc(mean_abs_shap))

agg_shap$orig_feature <- gsub("_", " ", agg_shap$orig_feature)

# Plot aggregate SHAP values (Top 10)
ggplot(agg_shap[1:10, ], aes(x = reorder(orig_feature, mean_abs_shap),, 
                             y = mean_abs_shap, fill = mean_abs_shap))+
  geom_col()+
  coord_flip()+
  scale_fill_gradient(low = "#c5cae9", high = "#4527a0")+
  labs(
    title = "Grouped SHAP Feature Importance (Top 10)", 
    x = "Feature",
    y = "Mean SHAP Value"
  )+
  theme_classic(base_size = 13)+
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# Compute SHAP Values for each feature level
#===============================================================================
level_shap_direction <- shap_long %>%
  group_by(variable) %>%
  summarise(
    mean_shap = mean(value), 
    mean_abs_shap = mean(abs(value)), 
    .groups = "drop"
  ) %>%
  arrange(desc(abs(mean_shap)))

ggplot(level_shap_direction, aes(x = reorder(variable, mean_shap), 
                                 y = mean_shap, fill = mean_shap))+
  geom_col()+
  coord_flip()+
  scale_fill_gradient2(low = "red", 
                       mid = "lightblue", 
                       high = "blue", midpoint = 0)+
  labs(title = "Mean SHAP Value per Feature Level (Direction & Magnitude)", 
       x = "Feature Level", 
       y = "Mean SHAP Value")+
  theme_minimal()

#===============================================================================
# Generate ROC Curve for XGBoost Model Performance
#===============================================================================

roc_obj <- roc(labels, predict(final_xgb_model, dtrain))

# Extract ROC points for plotting
roc_points <- coords(roc_obj, "all", ret = c("specificity", "sensitivity"))

roc_df_xgb <- data.frame(
  specificity = roc_points$specificity, 
  sensitivity = roc_points$sensitivity, 
  Model = "XGBoost"
)
# Plot ROC Curve
ggplot(roc_df_xgb, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue", lwd = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "ROC Curve for XGBoost Model Performance",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_classic() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# Generate ROC Curve per Fold XGBoost Model Performance
#===============================================================================

nfolds <- 10
labels <- getinfo(dtrain, "label")

# Create manual folds (stratified if desired)
folds <- caret::createFolds(labels, k = nfolds, 
                            list = TRUE, returnTrain = FALSE)

# Store true labels and predictions per fold 
fold_preds <- data.frame(
  fold = rep(NA, length(labels)),
  label = labels,
  pred = cv$pred
)

# Assign fold number to each observation
for (i in seq_along(folds)) {
  fold_preds$fold[folds[[i]]] <- i
}

# Extract ROC points for plotting
roc_data_xgb <- fold_preds%>%
  mutate(fold = as.factor(fold)) 

# Group by fold (resample), calculate ROC objects
roc_list <- fold_preds %>%
  group_by(fold) %>%
  summarise(roc_obj = list(roc(response = label, 
                               predictor = pred)), .groups = "drop")

# Extract ROC points for plotting
roc_points_xgb <- do.call(rbind, lapply(1:nrow(roc_list), function(i) {
  roc_obj <- roc_list$roc_obj[[i]]
  coords_df <- coords(roc_obj, "all", ret = c("specificity", "sensitivity"))
  data.frame(
    specificity = coords_df$specificity, 
    sensitivity = coords_df$sensitivity, 
    fold = roc_list$fold[i]
  )
}))

roc_points_xgb$fold <- as.factor(roc_points_xgb$fold)

# Plot ROC Curves for each fold
ggplot(roc_points_xgb, aes(x = 1 - specificity, y = sensitivity, color = fold)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "ROC Curve per Fold for XGBoost Model",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_classic() +
  theme(legend.position = "bottom") +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))


#===============================================================================
# Generate PR Curve for XGBoost Model Performance
#===============================================================================

# Predict probabilities for the positive class
predicted_probs <- predict(final_xgb_model, dtrain)

# Ensure labels are correct
actuals <- labels

# Calculate PR Curve 
pr <- pr.curve(scores.class0 = predicted_probs[actuals == 1], 
               scores.class1 = predicted_probs[actuals == 0], 
               curve = TRUE)

# Convert curve to a data frame
pr_df_xgb <- data.frame(Recall = pr$curve[, 1], 
                        Precision = pr$curve[, 2], 
                        Model = "XGBoost")

# Plot PR Curve
ggplot(pr_df_xgb, aes(x = Recall, y = Precision))+
  geom_line(color = "blue", size = 1)+
  labs(title = "Precision-Recall Curve", x = "Recall", y = "Precision")+
  theme_minimal()


#===============================================================================
# Create Confusion Matrix for XGBoost Model Performance
#===============================================================================

pred_probs <- predict(final_xgb_model, dtrain)

pred_classes <- ifelse(pred_probs > 0.5, "Yes", "No")

predictions <- factor(pred_classes, levels = c("No", "Yes"))
labels <- factor(training_data$Is_Revenue, levels = c("No", "Yes"))

# Compute Confusion Matrix
conf_matrix <- confusionMatrix(data = predictions,
                               reference = labels, 
                               positive = "Yes")

# Extract the table separately for display 
cm_table <- conf_matrix$table
cm_df <- as.data.frame(cm_table)
cm_df

# Rename columns 
colnames(cm_df) <- c("Predicted", "Actual", "Count")

# Create a nice-looking table for the confusion matrix
cm_df %>%
  gt() %>%
  tab_header(
    title = "Confusion Matrix"
  ) %>%
  cols_label(
    Predicted = "Predicted Class",
    Actual = "Actual Class",
    Count = "Number of Observations"
  ) %>%
  fmt_number(columns = c(Count), decimals = 0) %>%
  data_color(
    columns = c(Count),
    fn = scales::col_numeric(
      palette = c("white", "steelblue"),
      domain = NULL
    )
  )

# Create a table for additional performance metrics
precision <- as.numeric(conf_matrix$byClass["Precision"])
recall <- as.numeric(conf_matrix$byClass["Sensitivity"])
F1 <- 2 * (precision * recall)/ (precision + recall)

# Calculate ROC AUC
roc_obj <- roc(response = labels, predictor = pred_probs, 
               levels = c("No", "Yes"), direction = "<")
AUC <- auc(roc_obj)
print(AUC)

# Calculate PR AUC
pr_auc <- pr$auc.integral

metrics_xgb <- data.frame(
  Accuracy = conf_matrix$overall["Accuracy"],
  Recall = conf_matrix$byClass["Sensitivity"],
  Precision = conf_matrix$byClass["Precision"],
  F1 = F1, 
  ROC_AUC= roc_auc, 
  PR_AUC = pr_auc
  
)

print(metrics_xgb)

# Create a nice looking table for additional metrics
metrics_xgb %>%
  gt() %>%
  tab_header(
    title = "XGBoost Model Performance Metrics"
  ) %>%
  fmt_number(columns = everything(), decimals = 3) %>%
  cols_label(
    Accuracy = "Accuracy",
    Recall = "Recall",
    Precision = "Precision",
    F1 = "F1 Score", 
    ROC_AUC = "ROC AUC", 
    PR_AUC = "PR AUC"
    
  ) %>%
  data_color(
    columns = everything(),
    fn = scales::col_numeric(
      palette = c("#90caf9"),
      domain = c(0,1)
    )
  )

#===============================================================================
# Create Feature Importance plot for XGBoost Model 
#===============================================================================

importance_matrix <- xgb.importance(feature_names = colnames(features_matrix), 
                                    model = final_xgb_model)

importance_df_xgb <- as.data.frame(importance_matrix)

importance_df_xgb <- importance_df_xgb %>%
  filter(Gain > 0) %>%
  arrange(desc(Gain))

importance_df_xgb <- importance_df_xgb %>%
  head(10) %>%
  mutate(Feature = str_replace_all(Feature, "_", " ")) 

# Create a bar plot for feature importance
ggplot(importance_df_xgb, aes(x = reorder(Feature, Gain), 
                              y = Gain, fill = Gain)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance from XGBoost Model",
       x = "Features",
       y = "Importance Score") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_classic() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))
#===============================================================================
# Combined plot across all three models 
#===============================================================================

# Combine plots for feature performance comparison

glm_plot <- ggplot(coef_df, aes(x = Feature, y = Importance, fill = Importance))+
  geom_col()+
  coord_flip()+
  labs(title = "Feature Importance for Logistic Regression Model", 
       x = "Feature / Feature Level",
       y = "Absolute Coefficient")+
  scale_fill_gradient(low = "lightblue", high = "darkblue")+
  theme_classic()+
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))
rf_plot <- ggplot(importance_df_rf, aes(x = Feature, y = Overall, 
                                        fill = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance for Random Forest Model",
       x = "Feature / Feature Level",
       y = "Permutation Importance") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_classic() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))
xgboost_plot <- ggplot(importance_df_xgb, aes(x = Feature, 
                                              y = Gain, fill = Gain)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance for XGBoost Model",
       x = "Feature / Feature Level",
       y = "Gain Importance") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_classic() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

combine_plot <- glm_plot + rf_plot + xgboost_plot + plot_layout(ncol = 2)

# Display combined plot
print(combine_plot)

# Assign labels to the data for each model 
roc_df_glm <- roc_df_glm %>%mutate(Model = "Logistic regression")
roc_df_rf <- roc_df_rf %>%mutate(Model = "Random Forest")
roc_df_xgb <- roc_df_xgb %>%mutate(Model = "XGBoost")

# Combine all into a dataframe
all_roc <- bind_rows(roc_df_glm, roc_df_rf, roc_df_xgb)

# Plot combined ROC Curve 
combined_plot <- ggplot(all_roc, aes(x = 1 - specificity,
                                     y = sensitivity, color = Model))+
  geom_line(lwd = 1.2)+
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey70")+
  labs(
    title = "ROC Curve Comparison",
    subtitle = "Online Shopping Dataset", 
    x = "False Positive Rate (1 - Specificity)", 
    y = "True Positive Rate", 
    color = "Model"
  )+
  scale_color_brewer(palette = "Set1")+
  scale_x_continuous(limits = c(0,1), breaks= seq(0,1,0.2))+
  scale_y_continuous(limits = c(0,1), breaks= seq(0,1,0.2))+
  theme_minimal(base_size = 14)+
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom", 
    axis.title.y = element_text(angle = 0, vjust = 0.6), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black", size = 0.8)
    
  )

print(combined_plot)
 
# Combine performance metrics 
combined_metrics %>%
  gt() %>%
  tab_header(title = "Comparison of Model Performance Metrics") %>%
  fmt_number(columns = -Model, decimals = 3) %>%
  cols_label( 
    Model = "Model",
    Accuracy = "Accuracy",
    Recall = "Recall",
    Precision = "Precision",
    F1 = "F1 Score", 
    ROC_AUC = "ROC AUC", 
    PR_AUC = "PR AUC"
    
  ) %>%
  data_color(
    columns = c("Accuracy", "Recall", "Precision", "F1", "ROC_AUC", "PR_AUC"), 
    fn = scales::col_numeric(
      palette = c("#D0E3FA", "#2A5DAB"), 
      domain = c(0,1)
    )
  )

# Plot combined PR Curve
pr_combined <- bind_rows(pr_df_glm, pr_df_rf, pr_df_xgb)

ggplot(pr_combined, aes(x = Recall, y = Precision, color = Model))+
  geom_line(size = 1.2)+
  labs(title = "Precision-Recall Curve Comparison", 
       x = "Recall", y = "Precision")+
  theme_classic()+
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))+
  scale_color_manual(values = c(
    "Logistic Regression" = "#1E88E5", 
    "Random Forest" = "#33A02C", 
    "XGBoost" = "#E64A19"
  ))
