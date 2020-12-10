
# Bengali.AI Handwritten Grapheme Classification
https://www.kaggle.com/c/bengaliai-cv19/overview

41st place solution (out of 2,059 teams Top 2%)

private/public score of 0.9400/0.9849

# Approach

[1] First stage: train with Cutmix and Mixup augmentations for more than 100 epochs, as these augmentations require a big number of epochs to converge.

[2] Second stage: fine-tune for ~5 epochs without any aumentations at all.

[3] Ensemble over 5 folds using max voting technique.
