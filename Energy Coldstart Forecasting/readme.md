## Power Laws: Cold Start Energy Forecasting

https://www.drivendata.org/competitions/55/schneider-cold-start/submissions/

### Task specification

The objective of this competition is to forecast energy consumption from varying amounts of "cold start" data, and little other building information. That means that for each building in the test set given a small amount of data and then asked to predict into the future.

<img src=https://s3.amazonaws.com/drivendata-public-assets/mlscheme.png>

### Dataset

Three time horizons for predictions are distinguished. The goal is either:

* To forecast the consumption for each hour for a day (24 predictions).
* To forecast the consumption for each day for a week (7 predictions).
* To forecast the consumption for each week for two weeks (2 predictions).
In the test set, varying amounts of historical consumption and temperature data are given for each series, ranging from 1 day to 2 weeks. The temperature data contains a portion of wrong / missing values.

In the training set, 4 week series of hourly consumption and temperature data are provided. These series can be used to create different cold start regimes (varying amounts of provided data and prediction resolutions) for local training and testing.

Basic building information such as suface area, base temperature, and on/off days are given for each series in the training and test sets.

#### Historical Consumption

Time series data of consumption and temperature data identified by their series_id.

* series_id - An ID number for the time series, matches across datasets
* timestamp - The time of the measurement
* consumption - Consumption (watt-hours) since the last measurement
* temperature - Outdoor temperature (Celsius) during measurement from nearby weather stations, some values missing

#### Meta data

* series_id - An ID number for the time series, matches across datasets
* surface - The surface area of the building (ordinal)
* base_temperature - The base temperature that the inside of the building is set to (ordinal)
* monday_is_day_off - Whether or not the building is operational this day
* tuesday_is_day_off - Whether or not the building is operational this day
* wednesday_is_day_off - Whether or not the building is operational this day
* thursday_is_day_off - Whether or not the building is operational this day
* friday_is_day_off - Whether or not the building is operational this day
* saturday_is_day_off - Whether or not the building is operational this day
* sunday_is_day_off - Whether or not the building is operational this day

### Scoring

Normalized MAE (mean absolute error)

### Solution overview

1. <b> Regression models </b> - LGBM/XGB regression with meta information, lagged variables and encoding of timesteps
2. <b> Deep learning models </b> - LSTM, BiLSTM, Seq2seq, Seq2seq with attention
3. <b> Ensemble </b> - weighted average of LSTM models

### Final score

Final score of 0.407 with leaderboard rank 80 among 1290 participants. Highest score achieved is 0.258

### Other links

Winning solutions - https://github.com/drivendataorg/power-laws-cold-start/


