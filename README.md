
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
* Public test set size is only 12 images!

# Approach

## Final solution
* Three stage pipeline:
   1. First stage: 
       * Model: EfficeintNet B0-B3 with three heads
       * Head configuration Mish -> Conv2D -> BatchNorm -> Pooling layer -> Linear
       * Train with Cutmix (alpha=1) and Mixup (alpha=4) augmentations for more than 100 epochs, as these augmentations require a big number of epochs to converge.
       * Pooling layer: 0.5 * (AveragePooling + MaxPooling)
       * Dataset: 5-fold of uncropped images generated via stratified split 
       * Weights: 7-grapheme, 1-consonant, 2-vowel
       * AdamW and OneCycleWithWarmUp
       

   2. Second stage: 
      * fine-tune for ~5 epochs without any aumentations at all.
      * same dataset and architecture as in the first layer.

   3. Ensemble over 5 folds using max voting technique.

## Baseline solution

* Resnet-50 with 3 heads
* 30 epochs
* Basic gemetric augmentations: Scale, Rotate, HorizontalFlip, etc.
* Cross Entropy Loss
* Cropped and centered images
* Random split
* Result: 0.95 public leaderboards

## Things that didn't work
* se-resnexts and resnets
* OHEM loss
* 3 different models insted single model with 3 heads
* Basic geometric configurations

# Run
The dataset is available at Kaggle via this link: https://www.kaggle.com/c/bengaliai-cv19/data  
Or via Kaggle API: `https://www.kaggle.com/c/bengaliai-cv19/data`

You need to add your own path to the dataset in config files, which are located in `configs` folder. Specificaly you need to put your own values the following fields:
*  `into train_csv_path`
*  `train_csv_name`
*  `data_folder`

Assuming you are in the project root, then you can run the first stage for the first fold with:
`catalyst-dl run --expdir src --logdir {YOUR_PATH_TO_LOGDIR} --config configs/triple_head/mixup_cutmix_efnet_f0.yml`

Then run the second stage for the first fold with:
`catalyst-dl run --expdir src --logdir {YOUR_PATH_TO_LOGDIR} --config configs/triple_head/finetune_efnet_f0.yml`

After you've completed training for all 5 folds you can ensemble predictions using the code form this kernel:
https://www.kaggle.com/lightnezzofbeing/5-fold-average
