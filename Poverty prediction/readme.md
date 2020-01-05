## Pover-T Tests: Predicting Poverty

https://www.drivendata.org/competitions/50/worldbank-poverty-prediction/

### Task specification

The task is to predict whether or not a given household for a given country is poor or not. The training features are survey data from three countries. For each country, A, B, and C, survey data is provided at the household as well as individual level. Each household is identified by its id, and each individual is identified by both their household id and individual iid. Most households have multiple individuals that make up that household.

### Dataset

Data for each of the three countries is provided at the household and individual level. There are six training files in total.

Each column in the dataset corresponds with a survey question. Each question is either multiple choice, in which case each choice has been encoded as random string, or it is a numeric value. Many of the multiple choice questions are about consumable goods--for example does your household have items such as Bar soap, Cooking oil, Matches, and Salt. Numeric questions often ask things like How many working cell phones in total does your household own? or How many separate rooms do the members of your household occupy?

### Scoring

Average log loss

### Solution overview

1. <b> EDA </b>
2. <b> Feature engineering and selection </b> - features generated from anonymized categorical features. Boruta for feature selection.
3. <b> Modelling </b> - H2O autoML

### Final score

Score of 0.19 with leaderboard rank of 211 among 2300+. Highest score achieved is 0.147.

### Other links

Winning solutions - https://github.com/drivendataorg/pover-t-tests/
