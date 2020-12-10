
# Bengali.AI Handwritten Grapheme Classification
https://www.kaggle.com/c/bengaliai-cv19/overview

**41st place solution (out of 2,059 teams Top 2%)**

**private/public score of 0.9400/0.9849**

# Competition description

## Task formulation
You’re given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2Fa9a48686e3f385d9456b59bf2035594c%2Fdesc.png?generation=1576531903599785&alt=media" width="60%" height="60%" align="center">

## Data and metric 

* ~200000 single channel images
* Class distribution: 
    * Grapheme root: 168 classes
    * Vowel diacritic: 11 classes
    * Consonant diacritic: 7 classes
    * 168 * 11 * 77 ~ 13k different grapheme variations
* Metric: weighted sum of recall for each component with weights:
  * 2 - Grapheme root
  * 1 - Vowel diacritic
  * 1 - Consonant diacritic

# Approach

## Final solution
* Three stage pipeline:
   1. First stage: train with Cutmix and Mixup augmentations for more than 100 epochs, as these augmentations require a big number of epochs to converge.

   2. Second stage: fine-tune for ~5 epochs without any aumentations at all.

   3. Ensemble over 5 folds using max voting technique.

## Baseline solution

* Resnet-50
* One model with 3 heads
* 30 epochs
* Basic gemetric augmentations: Scale, Rotate, HorizontalFlip, etc.
* Cross Entropy Loss
* Cropped and centered images
* Random split
* Result: 0.95 public leaderboards
