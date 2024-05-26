import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Resnet50(nn.Module):
    def __init__(self, features_size=2048, out_size=14):
        super(Resnet50, self).__init__()
        # Load a pretrained ResNet-50 model
        weights = ResNet50_Weights.DEFAULT
        self.shared = resnet50(weights=weights)

        # Remove the last fully connected layer (classifier) to use as a feature extractor
        self.shared.fc = nn.Identity()

        # Define new heads with the specified output size
        self.head = nn.Linear(features_size, out_size)
        self.head2 = nn.Linear(features_size, out_size)
        self.head3 = nn.Linear(features_size, out_size)

    def forward(self, x, return_features=False):
        x = self.shared(x)
        if return_features:
            return self.head(x), self.head2(x), self.head3(x), x
        return self.head(x), self.head2(x), self.head3(x)
