library(data.table)
library(caret)
library(h2o)
library(dplyr)
library(plyr)
require(caTools)
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
  
  #sample = sample.split(train$poor, SplitRatio = .75)
  #train_val = subset(train, sample == TRUE)
  #val  = subset(train, sample == FALSE)
  
  train.hex <- as.h2o(train)
  test.hex <- as.h2o(test)
  
  split_h2o <- h2o.splitFrame(train.hex, c(0.8,.15))
  train_conv_h2o <- h2o.assign(split_h2o[[1]], "train")
  valid_conv_h2o <- h2o.assign(split_h2o[[2]], "valid")
  test_conv_h2o  <- h2o.assign(split_h2o[[3]], "test")
  
  target <- "poor"
  predictors <- setdiff(names(train_conv_h2o), target)
  
  aml <- h2o.automl(
    x = predictors,
    y = target,
    training_frame = train_conv_h2o,
    leaderboard_frame = valid_conv_h2o,
    max_runtime_secs = 3600 # change it to 3600
  )
  
  perf <- h2o.performance(aml@leader,test_conv_h2o)
  h2o.confusionMatrix(perf)
  
  model = aml@leader
  model_path <- h2o.saveModel(model, path=paste('model',s), force=TRUE)
  
  #ensemble <- h2o.stackedEnsemble(x = predictors, y = target, training_frame = as.h2o(train), base_models = list(model))
  
  pred <- h2o.predict(model, as.h2o(test))
  #h2o.table(pred$predict, test_conv_h2o$poor)
  test_pred <- data.frame(id = c(test_ids),country = c(country),poor = as.data.frame(pred)$True)
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

write.csv(submission1, 'submission2.csv', row.names = FALSE)
