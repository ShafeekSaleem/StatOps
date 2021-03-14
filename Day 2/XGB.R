library(ggplot2)
library(dplyr)
library(statsr)
library(xgboost)
library(MLmetrics)
library(lubridate)
library(corrplot)
library(truncnorm)
library(caret) 
library(gridExtra)
library(DMwR)
library(ROSE)

#------------------------------------------------------------------------------#
# Global variables
{
  model_name <- "xgb_v1"  # Create your own model name
  seed_value <- 1234
  
  lazy_load <- FALSE  # Load from RDS files if train/test sets are already created
  
  number_grids <- 5  # Number of iterations for random grid search
}

#------------------------------------------------------------------------------#

setwd("C:\\Users\\octavei\\OneDrive - John Keells Holdings PLC\\Desktop\\MyFiles\\DATASTORM")
test_with_id <- read.csv("testwithid.csv", na.strings = "")
reserv_id <- test_with_id$Reservation.id
#train <- read.csv("smote\\train.csv",na.strings = "")
train <- read.csv("TrainingCleaned.csv",na.strings = "")
validation <- read.csv("ValidationCleaned.csv",na.strings = "")
#validation <- read.csv("validation.csv",na.strings = "")
test <-test_with_id %>% select(-Reservation.id)
colnames(train)[1] <- "Age"
colnames(validation)[1] <- "Age"
colnames(test)[1] <- "Age"
# drop columns
x <- train %>% select(-Expected_checkin, -Expected_checkout, -Booking_date , -Adults, -Children, -Is_Latino,-East, -Resort, -Refundable, -Direct, -Rooms_Needed, -WeekDay ,-Use_Promotion, -Previous_Cancellations, -Airport.Hotels)
# #x <- train %>% select(-Adults, -Children, -Is_Latino,-East, -Resort, -Direct, -Rooms_Needed, -WeekDay ,-Use_Promotion, -Previous_Cancellations, -Airport.Hotels)
val <- validation %>% select(-Expected_checkin, -Expected_checkout, -Booking_date , -Adults, -Children, -Is_Latino,-East, -Resort,-Refundable, -Direct, -Rooms_Needed, -WeekDay, -Use_Promotion, -Previous_Cancellations, -Airport.Hotels)
final <- test %>% select(-Expected_checkin, -Expected_checkout, -Booking_date , -Adults, -Children, -Is_Latino,-East, -Resort,-Refundable, -Direct, -Rooms_Needed, -WeekDay, -Use_Promotion, -Previous_Cancellations, -Airport.Hotels)
#x <- train %>% select(-Expected_checkin, -Expected_checkout, -Booking_date)
#x <- train %>% select(-Adults, -Children, -Is_Latino,-East, -Resort, -Direct, -Rooms_Needed, -WeekDay ,-Use_Promotion, -Previous_Cancellations, -Airport.Hotels)
#val <- validation %>% select(-Expected_checkin, -Expected_checkout, -Booking_date)
#final <- test %>% select(-Expected_checkin, -Expected_checkout, -Booking_date )

# M <- cor(x) # get correlations
# corrplot(M, method = "square") 
temp <- x %>% select(-Reservation_Status)
mod_vars <- colnames(temp)

val <- val %>% select(mod_vars, Reservation_Status)
final <- final %>% select(mod_vars)
{
  # xx <- x
  # xx$Reservation_Status <- as.factor(xx$Reservation_Status)
  # xx %>% group_by(Reservation_Status) %>% summarise(count = n(),perc = n()*100/count(xx))
  # x_under <- ovun.sample(Reservation_Status ~., data = xx, method = "under")$data
  # 
  # x_smote <- SMOTE(Reservation_Status ~ ., x, perc.over = 1000)
  # x_smote %>% group_by(Reservation_Status) %>% summarise(count = n(),perc = n()*100/count(x_smote))
  
}
{
  x %>% group_by(Reservation_Status) %>% summarise(count = n(),perc = n()*100/count(x))
  val %>% group_by(Reservation_Status) %>% summarise(perc = n()/count(val))
  # Reservation_Status perc$n
  # <int>  <dbl>                    < weight >
  # 1                  0  0.586          0.2474
  # 2                  1  0.270          0.537
  # 3                  2  0.145          1
  
  x$weights <- 0.80
  x$weights[x$Reservation_Status==0] <- 0.101
  x$weights[x$Reservation_Status==1] <- 0.515
  
  # shuffle
  set.seed(42)
  rows <- sample(nrow(x))
  x_shuffled <- x[rows, ]
  
}
{
  target <- x_shuffled$Reservation_Status
  target_val <- val$Reservation_Status
  weight <- x_shuffled$weights
  # sparse_matrix <- model.matrix(Reservation_Status~.-1, data = x)
  # sparse_matrix_val <- model.matrix(Reservation_Status~.-1, data = val)
  
  x_feature <- x_shuffled %>% select(-Reservation_Status, -weights)
  val_feature <- val %>% select(-Reservation_Status)
}


### Train vs Test split
{
  ## 90% of the sample size
  # Make split index
  train_index <- sample(1:nrow(x_feature), nrow(x_feature)*0.8)

  # split train data and make xgb.DMatrix
  train_data   <- x_feature[train_index,]
  train_label  <- target[train_index]
  train_weight <- weight[train_index]

  dtrain <- xgb.DMatrix(data = as.matrix(train_data), label = train_label, weight = train_weight)
  # split test data and make xgb.DMatrix
  test_data  <- x_feature[-train_index,]
  test_label <- target[-train_index]
  dtest <- xgb.DMatrix(data = as.matrix(test_data), label = test_label)
  watchlist <- list(train=dtrain, test=dtest)
  
  # full data
  dMatrix <- xgb.DMatrix(data = as.matrix(x_feature), label=target, weight = weight)
  # val data
  dVal <- xgb.DMatrix(data = as.matrix(val_feature), label= target_val)
  # test data
  dFinal <- xgb.DMatrix(data = as.matrix(final))
}


{
  numberOfClasses <- 3
  xgb_params <- list("objective" = "multi:softprob",
                     "eval_metric" = "mlogloss",
                     "num_class" = 3)
  nround    <- 100 # number of XGBoost rounds
  cv.nfold  <- 5
  
  # Fit cv.nfold * cv.nround XGB models and save OOF predictions
  cv_model <- xgb.cv(params = xgb_params,
                     data = dtrain, 
                     nrounds = nround,
                     nfold = cv.nfold,
                     verbose = TRUE,
                     prediction = TRUE, print_every_n = 10)
}
{
  OOF_prediction <- data.frame(cv_model$pred) %>%
    mutate(max_prob = max.col(., ties.method = "last"),
           label = train_label + 1)
  head(OOF_prediction,50)
}
{
  # confusion matrix
  confusionMatrix(factor(OOF_prediction$max_prob),
                  factor(OOF_prediction$label),
                  mode = "everything")
}

{
  bst_model <- xgb.train(params = xgb_params,
                         data = dMatrix,
                         nrounds = nround)
  
  # Predict hold-out test set
  test_pred <- predict(bst_model, newdata = dVal )
  test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                            ncol=length(test_pred)/numberOfClasses) %>%
    t() %>%
    data.frame() %>%
    mutate(label = target_val + 1,
           max_prob = max.col(., "last"))
  # confusion matrix of test set
  confusionMatrix(factor(test_prediction$max_prob),
                  factor(test_prediction$label),
                  mode = "everything")
}
{
  # prediction for test data
  final_test_pred <- predict(bst_model, newdata = dFinal )
  final_test_prediction <- matrix(final_test_pred, nrow = numberOfClasses,
                            ncol=length(final_test_pred)/numberOfClasses) %>%
    t() %>%
    data.frame() %>%
    mutate("Reservation-id" = reserv_id, Reservation_status = max.col(., "last")) %>% select("Reservation-id", Reservation_status)
  write.csv(final_test_prediction,"submission.csv" )
  
  }

{
  # compute feature importance matrix
  importance_matrix = xgb.importance(feature_names = vars, model = bst_model)
  head(importance_matrix,28)
}













{
  numberOfClasses <- length(unique(dat$Site))
  xgb_params <- list("objective" = "multi:softprob",
                     "eval_metric" = "mlogloss",
                     "num_class" = numberOfClasses)
  nround    <- 50 # number of XGBoost rounds
  cv.nfold  <- 5
  
  # Fit cv.nfold * cv.nround XGB models and save OOF predictions
  cv_model <- xgb.cv(params = xgb_params,
                     data = train_matrix, 
                     nrounds = nround,
                     nfold = cv.nfold,
                     verbose = FALSE,
                     prediction = TRUE)
  
}

#------------------------------------------------------------------------------#
# Random grid search hyper-parameter tuning
{
  df_param <- NULL
  vec_seeds <- NULL
  best_metric <- NULL
  best_iter <- NULL
  list_models <- list()
  
  set.seed(seed_value)
}


param <- list(objective = "multi:softprob",
              eval_metric = "mlogloss",
              max_depth = 5,
              # eta = round(runif(1, 0.01, 0.2), 4),
              eta = 0.01,
              # gamma = round(runif(1, 1, 2), 4),
              subsample = 0.7,
              #colsample_bytree = 0.85,
              min_child_weight = 100,
              colsample_bylevel = 1,
              lambda = 1,
              num_class = 3)

xgb_seed <- sample.int(1000, 1)[[1]]
set.seed(xgb_seed)

model <- xgb.train(params = param,
                       data = dMatrix,
                       nrounds = 200,
                       verbose = TRUE,
                       #early_stopping_rounds = 50,
                       watchlist = list(tst = dtest),
                       print_every_n = 10)

pred <- predict(model, dMatrix)
pred_test <- predict(model, dVal)

# get & print the classification error
acc <- mean(as.numeric(pred > 0.5) == target)
acc_test <- mean(as.numeric(pred_test > 0.5) == target_val)

print(paste("train-accuracy=", round(acc,4)*100,"%"))
print(paste("test-accuracy=", round(acc_test,4)*100,"%"))


importance_matrix <- xgb.importance(model = model)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)









for (iter in 1:50) {
  
  param <- list(objective = "multi:softprob",
                eval_metric = "mlogloss",
                max_depth = sample(3:20, 1),
                # eta = round(runif(1, 0.01, 0.2), 4),
                eta = sample(c(0.02, 0.05, 0.1, 0.15, 0.01), 1),
                # gamma = round(runif(1, 1, 2), 4),
                subsample = round(runif(1, 0.69, 0.71), 4),
                colsample_bytree = sample(seq(0.7, 1, 0.05), 1),
                min_child_weight = sample(seq(20, 100, 10), 1),
                colsample_bylevel = 1,
                lambda = 1,
                num_class = 3,
                alpha = sample(seq(0.7, 1, 0.05), 1))
  
  xgb_seed <- sample.int(1000, 1)[[1]]
  cat("\nRandom search ", iter, "..\n", sep = "")
  cat("Seed ", xgb_seed, "\n", sep = "")
  print(unlist(param))
  set.seed(xgb_seed)
  
  tmp_model <- xgb.train(params = param,
                         data = dMatrix,
                         nrounds = 100,
                         verbose = TRUE,
                         early_stopping_rounds = 50,
                         watchlist = list(tst = dtest),
                         print_every_n = 10)
  
  df_param <- rbind(df_param, unlist(param))
  vec_seeds <- c(vec_seeds, xgb_seed)
  best_iter <- c(best_iter, tmp_model$best_iteration)
  best_metric <- c(best_metric, tmp_model$best_score)
  
  list_models[[iter]] <- tmp_model
}

# Save tuning results
{
  res_hyperpar <- data.frame(df_param, stringsAsFactors = F)
  res_hyperpar$xgb_seed <- vec_seeds
  res_hyperpar$best_iter <- best_iter
  res_hyperpar$best_metric <- best_metric
  
  res_hyperpar %>% View
  
  write.csv(res_hyperpar, paste0(dir_str, "/", job_str, "_gridsearch.csv"), row.names = F)
  saveRDS(list_models, paste0(dir_str, "/", job_str, "_gridmodels.RDS"))
}

# Get best hyperparameters
{
  idx_best <- which.max(res_hyperpar$best_metric)
  best_par <- res_hyperpar[which.max(res_hyperpar$best_metric), ]
  best_model <- list_models[[idx_best]]
  saveRDS(list_models, paste0(dir_str, "/", job_str, "_bestmodel.RDS"))
  saveRDS(best_par, paste0(dir_str, "/", job_str, "_besthyper.RDS"))
  
  cat("\nBest parameters:\n")
  print(best_par)
}







