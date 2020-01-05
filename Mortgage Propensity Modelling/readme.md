## Propensity to Fund Mortgages

https://www.crowdanalytix.com/contests/propensity-to-fund-mortgages

### Task description

Propensity to fund mortgages describes the natural tendency for a mortgage to be funded based on certain factors in a customer’s application data.

To predict whether a mortgage will be funded using only this application data, certain leading factors driving the loan’s ultimate status will be identified. Solvers will discover the specific aspects of the dataset that have the greatest impact, and build a model based on this information.

### Dataset description

There are 20 fields in the dataset, which represent a past customer’s mortgage application data. 

### Scoring

Macro F1 score

### Solution overview

1. <b> EDA and exploratory analysis </b> - Hypothesis testing 
2. <b> Feature engineering </b> - categorical column encoding, quantization of continuous columns into categorical, clustering and auto encoder features
3. <b> Modelling </b> - Catboost, LightGBM
4. <b> Ensemble </b> - Blending, Stacking, Meta modelling
5. <b> AutoML </b> - H2O AutoML, tpot
6. <b> Model explaination </b> - Shap analysis

### Final score

0.65 with leaderboard rank 69 among 450+ participants. Highest F1 score achieved 0.75
