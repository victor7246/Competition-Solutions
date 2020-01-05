library(data.table)
library(caret)
library(h2o)
library(dplyr)
library(plyr)
require(caTools)
library(ROCR)
library(xgboost)

set.seed(101) 

h2o.init()

total_pred <- c()

create_prediction <- function(s){
  print(s)
  train <- read.csv(paste(s,'_train_full.csv',sep=""))
  train_ids <- train$id
  train <- subset(train, select = -c(id,iid))
  train[,'poor'] = as.factor(train[,'poor'])
  test <- read.csv(paste(s,'_test_full.csv',sep=""))
  test_ids <- test$id
  country <- rep(s,length(test_ids))
  test <- subset(test, select = -c(id,iid,poor))
  
  print(nrow(train))
  print(nrow(test))
  print(ncol(train))
  print(ncol(test))
  
  sample = sample.split(train$poor, SplitRatio = .75)
  train_val = subset(train, sample == TRUE)
  val  = subset(train, sample == FALSE)
  
  #model <- glm (poor ~ ., data = train_val, family = binomial)
  
  #summary(model)
  
  #predict <- predict(model, type = 'response')
  
  #predict1 <- predict(model, val)
  
  xgb <- xgboost(data = data.matrix(subset(train,select = -c(poor))), 
                               label = train$poor == 'True', 
                                 eta = 0.08,
                                 max_depth = 12, 
                                 nround=150, 
                                lambda = .5,
                                 subsample = 0.85,
                                colsample = .8,
                                 seed = 12345,
                                 eval_metric = "logloss",
                                 objective = "binary:logistic"
                  )
  
  #predict2 <- predict(xgb, data.matrix(subset(val, select = -c(poor))))
  
  #predict <- predict(xgb, type = 'response')
  
  #table(val$poor, predict1 > 0.5)
  #table(val$poor, predict2 > 0.5)
  
  #pred <- h2o.predict(model, as.h2o(test))
  pred <- predict(xgb, data.matrix(test))
  
  #h2o.table(pred$predict, test_conv_h2o$poor)
  test_pred <- data.frame(id = c(test_ids),country = c(country),poor = pred)
  test_pred <- ddply(test_pred, "id", summarise, poor = max(poor), country = s)
  
  return(test_pred)
  
}  
test_pred <- create_prediction('A')
total_pred <- rbind(total_pred,test_pred)
test_pred <- create_prediction('B')
total_pred <- rbind(total_pred,test_pred)
test_pred <- create_prediction('C')
total_pred <- rbind(total_pred,test_pred)

submission_file <- read.csv('submission_format.csv')
submission_file <- subset(submission_file, select = -c(poor))
nrow(submission_file)
length(unique(submission_file$id))
total_pred <- subset(total_pred, select = -c(country))
nrow(total_pred)
length(unique(total_pred$id))
submission1 <- merge(submission_file,total_pred,all.x = TRUE,sort = FALSE)

write.csv(submission1, 'submission7.csv', row.names = FALSE)
