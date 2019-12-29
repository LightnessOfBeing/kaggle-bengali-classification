from torch import nn

# we have multiple criterions
criterion = {
    "ce": nn.CrossEntropyLoss(),
    # Define your awesome losses in here. Ex: Focal, lovasz, etc
}