## Bengali.AI Handwritten Grapheme Classification

https://www.kaggle.com/c/bengaliai-cv19

### Task specification

The objective of this contest is as follows:

* Given an image of a handwritten Bengali grapheme, classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.

<img src=https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2Fa9a48686e3f385d9456b59bf2035594c%2Fdesc.png?generation=1576531903599785&alt=media>

### Dataset

The data set consists of the following fields:

train.csv
* image_id: the foreign key for the parquet files
* grapheme_root: the first of the three target classes
* vowel_diacritic: the second target class
* consonant_diacritic: the third target class
* grapheme: the complete character. Provided for informational purposes only, you should not need to use this.

train.parquet
Each parquet file contains tens of thousands of 137x236 grayscale images. The images have been provided in the parquet format for I/O and space efficiency. Each row in the parquet files contains an image_id column, and the flattened image.

### Scoring

Hierarchical macro-averaged recall

### Solution overview

1. <b> Image classification models </b> - EfficientNet-b3, EfficientNet-b6, Densenet
2. <b> Augmentation </b> - mixup
3. <b> Ensemble </b> - Stacking

### Final score

Recall score of 0.9297 (rank 233) out of 2000 teams. (highest score 0.9762)

### Resources

Top solutions - https://www.kaggle.com/c/bengaliai-cv19/discussion/136102
Augmentation - https://www.kaggle.com/ipythonx/keras-grapheme-gridmask-augmix-ensemble
Additional techniques - SSE pooling, GEM pooling, Arcface detection for unseen images, CAMCutmix/ZoneCutmix
Additional Links -	
https://www.kaggle.com/c/bengaliai-cv19/discussion/136815
https://www.kaggle.com/c/bengaliai-cv19/discussion/134905
https://www.kaggle.com/c/bengaliai-cv19/discussion/136030

