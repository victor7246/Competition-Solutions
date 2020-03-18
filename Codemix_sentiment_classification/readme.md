## SentiMix Hindi-English

### Task Specifications
Mixing languages, also known as code- mixing, is a norm in multilingual societies. The task is to predict the sentiment of a given code-mixed tweet. The sentiment labels are positive, negative, or neutral.

### Dataset

* The data is in CONLL format. It looks like:

meta    uid    sentiment 
token    lang_id 

 * Uid is a unique id for each tweet. lang_id is 'HIN' if the token is in Hindi, 'ENG' if the token is in English, and 'O' if the token is in neither of the languages. 

### Scoring

Average macro F1
  
### Solution overview

1. <b> Modelling </b> - BiLSTM with attention at subword/character/word level, Transformer, Bert

### Final score

Final macro F1 score achieved on validation data is 0.63




