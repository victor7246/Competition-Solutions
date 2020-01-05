## Gamma Log Facies Type Prediction

https://www.crowdanalytix.com/contests/gamma-log-facies-type-prediction

### Task specification

The objective of this contest is as follows:

* Given an array of GR (Gamma-Ray) values, accurately predict the log facies type corresponding to each value

<img src=http://www.sepmstrata.org/CMS_Images/EMERYTRENDS-FINAL.JPG>

### Dataset

The data set consists of the following fields:

* well_id & row_id: columns uniquely identifying each row
* GR: Gamma-Ray log values
* label: Variable to be predicted

### Scoring

Multiclass accuracy score

### Solution overview

1. <b> Seq2seq models </b> - Bidirectional LSTM, CNN-BiLSTM, BiLSTM with self attention, BiLSTM with positional encoding
2. <b> Ensemble </b> - Stacking

### Final score

0.968 with leaderboard rank of 26 among 350+ participants. Highest score achieved is 0.993
