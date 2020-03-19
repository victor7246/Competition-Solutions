## Google QUEST Q&A Labeling

https://www.kaggle.com/c/google-quest-challenge

### Task specification

The objective of this contest is as follows:

* The objective of the competition was to build a predictive model for different subjective aspects of question-answering systems.

<img src=https://storage.googleapis.com/kaggle-media/competitions/google-research/human_computable_dimensions_1.png>

### Dataset

The data for this competition includes questions and answers from various StackExchange properties. The task is to predict target values of 30 labels for each question-answer pair.

Each row contains a single question and a single answer to that question, along with additional features. Target labels are aggregated from multiple raters, and can have continuous values in the range [0,1]. 

### Scoring

Mean column-wise Spearman's correlation coefficient

### Solution overview

1. <b> Feature engineering </b> - basic text features, word embedding, universal sentence embedding features, topic model/decomposition based features
2. <b> Model </b> - Bert base
3. <b> Postprocessing </b>

### Final score

Final score of 0.38469 (rank 116) out of 1500 teams. (highest score 0.431)

### Resources

Top solutions - 

https://www.kaggle.com/c/google-quest-challenge/discussion/130047

https://www.kaggle.com/c/google-quest-challenge/discussion/131103

https://www.kaggle.com/jionie/models-with-optimization-v5

