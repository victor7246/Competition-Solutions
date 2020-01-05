
# coding: utf-8

# In[1]:

#get_ipython().magic('matplotlib inline')

# plotting
import matplotlib as mpl
mpl.style.use('ggplot')
import matplotlib.pyplot as plt

# math and data manipulation
import numpy as np
import pandas as pd
import pickle

# set random seeds 
from numpy.random import seed
from dateutil.parser import parse
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# modeling
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Bidirectional, GRU, Flatten, Dropout

# progress bar
from tqdm import tqdm


# In[2]:

train = pd.read_csv('../data/consumption_train.csv',parse_dates=['timestamp'],index_col=0)
train.head(5)


# In[3]:

print (train.shape)
print (train.series_id.nunique())


# In[4]:

test = pd.read_csv('../data/cold_start_test.csv',index_col=0,parse_dates=['timestamp'])
test.head(5)


# In[5]:

print (test.shape)
print (test.series_id.nunique())
print (test.series_id.value_counts().describe())



# In[14]:

def create_lagged_features(df, lag=1):
    if not type(df) == pd.DataFrame:
        df = pd.DataFrame(df, columns=['consumption'])
    
    def _rename_lag(ser, j):
        ser.name = ser.name + f'_{j}'
        return ser
        
    # add a column lagged by `i` steps
    for i in range(1, lag + 1):
        df = df.join(df.consumption.shift(i).pipe(_rename_lag, i))

    df.dropna(inplace=True)
    return df


# In[15]:

def prepare_training_data(consumption_series, lag):
    """ Converts a series of consumption data into a
        lagged, scaled sample.
    """
    # scale training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    consumption_vals = scaler.fit_transform(consumption_series.values.reshape(-1, 1))
    
    # convert consumption series to lagged features
    consumption_lagged = create_lagged_features(consumption_vals, lag=lag)
    
    cols = list(consumption_lagged.columns)
    cols.remove('consumption')
    # X, y format taking the first column (original time series) to be the y
    X = consumption_lagged.drop('consumption', axis=1).values
    y = consumption_lagged.consumption.values
    
    # keras expects 3 dimensional X
    X = X.reshape(X.shape[0], 1, X.shape[1])
    #X = X.reshape(X.shape[0], X.shape[1])
    
    return X, y, scaler


# In[20]:

def create_lstm_model(lag):

    # model parameters
    batch_size = 1  # this forces the lstm to step through each time-step one at a time
    batch_input_shape=(batch_size, 1, lag)

    model = Sequential()


    model.add(LSTM(lag, return_sequences=True,stateful=True, batch_input_shape = batch_input_shape))
    #model.add(LSTM(24, return_sequences=True,stateful=True, activation='tanh'))
    model.add(LSTM(lag//2, stateful=True, activation='linear'))

    # followed by a dense layer with a single output for regression
    model.add(Dense(1))

    # compile
    model.compile(loss='mean_absolute_error', optimizer='adam')
    
    print (model.summary())
    
    return model


# In[21]


# In[ ]:

num_training_series = train.series_id.nunique()
epoch = 1
batch_size = 1
count = 0
model_dicts = {}
# reset the LSTM state for training on each series
for ser_id, ser_data in train.groupby('series_id'):

    #series_length = ser_data.shape[0]//24
    
    #if series_length%7 == 0:
    #    series_length = 7
    #else:    
    #    series_length = series_length%7
    
    for series_length in range(1,8):
        lag = 24*series_length
        if series_length in model_dicts:
            model = model_dicts[series_length]
        else:
            model = create_lstm_model(lag=lag)
        
        # prepare the data
        X, y, scaler = prepare_training_data(ser_data.consumption, lag)
    
        # fit the model: note that we don't shuffle batches (it would ruin the sequence)
        # and that we reset states only after an entire X has been fit, instead of after
        # each (size 1) batch, as is the case when stateful=False
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        
        model_dicts[series_length] = model
    
    count += 1
    if count%100 == 0:
        print ("count = {}".format(count))



# In[96]:
#pickle.dump(model_dicts, open('../all_lagged_models.pkl','wb'))
for i in model_dicts:
    model_dicts[i].save('../model_{}.h5'.format(i))

# In[97]:
def mape(y_true,y_pred):
    l = []
    for i in range(len(y_true)):
        if y_true[i] != 0:
            l.append(np.abs(y_true[i]-y_pred[i])/y_true[i]*100)
    return np.mean(l)

def generate_test_accuracy(model_dicts):
    y_final = []
    yhat_final = []
    global test
    for ser_id, ser_data in test.groupby('series_id'):
        series_length = ser_data.shape[0]//24
    
        if series_length%7 == 0:
            series_length = 7
        else:    
            series_length = series_length%7

        lag = 24*series_length
        if series_length in model_dicts:
            model = model_dicts[series_length]
        else:
            model = create_lstm_model(lag=lag)
        
        # prepare the data
        try:
            train2 = ser_data.iloc[:-24]
            test2 = ser_data.iloc[-24:]
            train_X, train_y, scaler = prepare_training_data(train2.consumption, lag)
            test_X, test_y, scaler2 = prepare_training_data(test2.consumption, lag)
            model.fit(train_X, train_y, epochs=1, batch_size=1, verbose=0, shuffle=False)
            y = scaler2.inverse_transform(test_y.reshape(-1, 1)).ravel()
            yhat = [0]*test_X.shape[0]
            for i in range(test_X.shape[0]):
                yhat[i] = model.predict(test_X[i,:,:].reshape(1, 1, lag), batch_size=1)[0][0]
            yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, 1)).ravel()  
            y_final += list(y)
            yhat_final += list(yhat)
        except:
            pass

    return np.array(y_final),np.array(yhat_final)    


# In[98]:

y,yhat = generate_test_accuracy(model_dicts)


# In[101]:

mape(y,yhat)


# In[ ]:

'''
### Results

with (0,1) scaling
1. 2 layers LSTM (24,12) =========> .465 => submission1
2. 3 layers LSTM (24,12,6) =======> .53 => submission2
3. 2 layers LSTM (12,6) with lag 12 ======> .408 => submission4
4. 2 layers LSTM (24,24) with lag 24 ======> .61 => submission5
5. 3 layers LSTM (24,24,12) with lag 24 ======> .39 => submission6 
6. 7 layers LSTM (24,24,18,12) with lag 24 ======>  => submission7 

4. 2 layers LSTM (12,6) with lag 12 bidirectional ======> .43 => submission3
5. 2 layers LSTM (24,12) with lag 24 bidirectional =====> 1.3
6. 2 layers LSTM (6,3) with lag 6 bidirectional =====> .45
7. 2 layers LSTM (12,12) with lag 12 bidirectional =====> .42
8. 2 layers LSTM (12,12) with lag 12 bidirectional with dropout =====>
9. 2 layers LSTM (12,12,6) with lag 12 bidirectional =====>
10. 2 layers LSTM (12,12,6) with lag 12 bidirectional with dropout =====>

with (-1,1) scaling
11. 2 layers LSTM (12,6) with lag 12 ======> .526
11. 2 layers LSTM (12,12) with lag 12 =========> 
12. 3 layers LSTM (24,12,6) =======> 
14. 2 layers LSTM (12,6) with lag 12 bidirectional ======> 
15. 2 layers LSTM (6,3) with lag 6 bidirectional =====>

'''
# In[33]:

def generate_hourly_forecast(num_pred_hours, consumption, model, scaler, lag):
    """ Uses last hour's prediction to generate next for num_pred_hours, 
        initialized by most recent cold start prediction. Inverts scale of 
        predictions before return.
    """
    # allocate prediction frame
    preds_scaled = np.zeros(num_pred_hours)
    
    # initial X is last lag values from the cold start
    X = scaler.transform(consumption.values.reshape(-1, 1))[-lag:]
    
    # forecast
    for i in range(num_pred_hours):
        # predict scaled value for next time step
        yhat = model.predict(X.reshape(1, 1, lag), batch_size=1)[0][0]
        preds_scaled[i] = yhat
        
        # update X to be latest data plus prediction
        X = pd.Series(X.ravel()).shift(-1).fillna(yhat).values

    # revert scale back to original range
    hourly_preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    return hourly_preds


# In[34]:

my_submission = pd.read_csv('../data/submission_format.csv')
my_submission.head(3)


# In[35]:

pred_window_to_num_preds = {'hourly': 24, 'daily': 7, 'weekly': 2}
pred_window_to_num_pred_hours = {'hourly': 24, 'daily': 7 * 24, 'weekly': 2 * 7 * 24}

num_test_series = my_submission.series_id.nunique()

for ser_id, pred_df in tqdm(my_submission.groupby('series_id'), 
                            total=num_test_series, 
                            desc="Forecasting from Cold Start Data"):
    
        
    # get info about this series' prediction window
    pred_window = pred_df.prediction_window.unique()[0]
    num_preds = pred_window_to_num_preds[pred_window]
    num_pred_hours = pred_window_to_num_pred_hours[pred_window]
    
    # prepare cold start data
    series_data = test[test.series_id == ser_id].consumption
    
    series_length = series_data.shape[0]//24
    
    if series_length%7 == 0:
        series_length = 7
    else:    
        series_length = series_length%7

    lag = 24*series_length
    if series_length in model_dicts:
        model = model_dicts[series_length]
    else:
        model = create_lstm_model(lag=lag)
    
    model.reset_states()
    
    cold_X, cold_y, scaler = prepare_training_data(series_data, lag)
    
    # fine tune our lstm model to this site using cold start data    
    model.fit(cold_X, cold_y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    
    # make hourly forecasts for duration of pred window
    preds = generate_hourly_forecast(num_pred_hours, series_data, model, scaler, lag)
    
    # reduce by taking sum over each sub window in pred window
    reduced_preds = [pred.sum() for pred in np.split(preds, num_preds)]
    
    # store result in submission DataFrame
    ser_id_mask = my_submission.series_id == ser_id
    my_submission.loc[ser_id_mask, 'consumption'] = reduced_preds

# In[36]:

my_submission.head(5)


# In[37]:

my_submission.to_csv("../data/submmission7.csv",index=False)


# In[ ]:



