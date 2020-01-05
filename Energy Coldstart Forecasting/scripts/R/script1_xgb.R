library(data.table)
library(caret)
library(h2o)
library(dplyr)
library(plyr)
require(caTools)
library(ROCR)
library(xgboost)
library(rBayesianOptimization)
#library(Boruta)

train = read.csv('../data/train_lagged.csv')
train = xgb.DMatrix(as.matrix(train[,!(names(train) %in% c('target','series_id'))]), label = as.matrix(train$target))

########### select features based on Boruta  #################################
#boruta.train <- Boruta(target~.-target, data = train, doTrace = 2)
#print(boruta.train)
#final.boruta <- TentativeRoughFix(boruta.train)
#cols = getSelectedAttributes(final.boruta, withTentative = F)

############### modeling #################

'''
params <- list(
  "eta"               = 0.02,
  "gamma"             = 1.45,
  "alpha" = .5,
  "lambda" = .5,
  "colsample_bytree" = .054,
  "max_depth" = 22,
  "min_child_weight" = 57,
  "objective"         = "reg:linear",
  "eval_metric"       = "rmse"
)
'''
params <- list(
  "eta"               = 0.1,
  "objective"         = "reg:linear",
  "eval_metric"       = "mae"
)

nfold = 5
nrounds = 2000


cv_scores = xgb.cv(data=train, params = params, nthread=4, maximize = FALSE,
                   nfold=nfold, nrounds=nrounds,print_every_n = 50,early_stopping_rounds = 15,
                   verbose = T)

model.xgb <- xgb.train(params=params,
                       data=train,
                       nrounds = nrounds,
                       watchlist=list(train=train),
                       print_every_n=50,
                       early_stopping_rounds = 15,verbose = 1)

test = read.csv('../data/test_lagged.csv')
y_true = test$target
test = xgb.DMatrix(as.matrix(test[,!(names(test) %in% c('target','series_id'))]))

y_pred = predict(model.xgb, test)

error <- c()
for (i in c(1:length(y_true)))
{
  if (y_true[i] != 0)
    error = c(error,abs(y_pred[i] - y_true[i])/y_true[i] * 100)
}  

mean(error)

submission = read.csv('../output/submission1.csv')

submission$target = expm1(pred)

write.csv(submission,'../output/submission_xgb2_0716.csv',row.names = FALSE)
