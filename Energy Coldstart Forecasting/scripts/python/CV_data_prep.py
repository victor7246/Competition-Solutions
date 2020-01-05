# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:53:56 2018

@author: asengup6
"""
import pandas as pd
import random

train = pd.read_csv('../data/consumption_train.csv')
test_train_new = pd.DataFrame()
test_new = pd.DataFrame()
test_actual = pd.DataFrame()
for series_id, data in train.groupby('series_id'):
    x = random.randint(1,8)
    train_new = pd.concat([train_new, data.iloc[:14*24]], axis=0)
    test_train_new = pd.concat([test_train_new, data.iloc[14*24:(28-x)*24]], axis=0)
    test_new = pd.concat([test_new, data[["series_id","timestamp","temperature","consumption"]].iloc[(28-x)*24:]] ,axis=0)
    test_actual = pd.concat([test_actual, data["consumption"].iloc[(28-x)*24:]], axis=0)
    
test_new['pred_id'] = range(test_new.shape[0])
test_new['prediction_windows'] = 'hourly'

test_new = test_new[["pred_id","series_id","timestamp","temperature","consumption","prediction_windows"]]
test_new.columns = ["pred_id","series_id","timestamp","temperature","consumption","prediction_window"]
test_new.consumption = 0
test_actual.columns = ['actual']
train_new.to_csv('../data/train_new.csv',index=False)
test_new.to_csv('../data/test_new.csv',index=False)
test_train_new.to_csv('../data/test_train_new.csv',index=False)
test_actual.to_csv('../data/test_actual.csv',index=False)