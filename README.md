
# Bengali.AI Handwritten Grapheme Classification
https://www.kaggle.com/c/bengaliai-cv19/overview

**41st place solution (out of 2,059 teams Top 2%)**

**private/public score of 0.9400/0.9849**

# Competition description
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2Fa9a48686e3f385d9456b59bf2035594c%2Fdesc.png?generation=1576531903599785&alt=media" width="60%" height="60%" align="center">

# Approach

[1] First stage: train with Cutmix and Mixup augmentations for more than 100 epochs, as these augmentations require a big number of epochs to converge.

[2] Second stage: fine-tune for ~5 epochs without any aumentations at all.

[3] Ensemble over 5 folds using max voting technique.
