## MAFAT Challenge - Fine-Grained Classification of Objects from Aerial Imagery

https://competitions.codalab.org/competitions/19854

### Dataset

The dataset consists of aerial imagery taken from diverse geographical locations, different times, resolutions, area coverage and image acquisition conditions (weather, sun direction, camera direction, etc). Image resolution varies between 5cm to 15cm GSD (Ground Sample Distance).

<img src=https://s3-us-west-2.amazonaws.com/codalab-webiks/Images/examples.jpg> </img>

### Task Specifications
Participants are asked to classify objects in four granularity levels:

Class - every object is categorized into one of the following major classes: 'Large Vehicles' or 'Small Vehicles'.

Subclass - objects are categorized to subclasses according to their function or designation, for example: Cement mixer, Crane truck, Prime mover, etc. Each object should be assigned to a single subclass.

Presence of features - objects are labeled according to their characteristics. For example: has a Ladder? is Wrecked? has a Sunroof? etc. Each object may be labeled with multiple different features

Object perceived color - Objects are labeled with their (human) percieved color.  For example: Blue, Red, Yellow etc. Each object includes a single color value.

### Scoring

For each label (each class, each subclass, each feature and each perceived color), an average precision index is calculated separately. Then, a Quality Index is calculated as the average of all average precision indices (Mean Average Precision).


```math
AP(label) = $ \frac{1}{K} \sum_{k=1}^{n}{Precision(k}\delta_k $
```
K is the total number of test images with true positive label, n is the total number of test images.

delta_k = 1 if object k is labelled with label, else 0.
Precision(k) is precision calculated over first k objects.
  
### Solution overview

1. Run scripts/crop_tiles_train.py to crop images and create numpy array
2. Use notebooks/baseline_mobilenet.ipynb 
3. Use notebooks/data_generator.ipynb to augment rare classes

### Final score

MAP score of 0.55 with leaderboard rank of 16. Highest MAP score achieved is 0.63.
