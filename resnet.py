import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet50(nn.Module):
    def __init__(self, model, num_labels):
        super(Resnet50, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            # model.fc,
        )
        self.layer4 = model.layer4
        self.num_labels = num_labels
        self.linear = nn.Linear(1000, self.num_labels)
        self.linear2 = nn.Linear(in_features=2048, out_features=1000, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear2(x)
        x = self.linear(x)
        return x


def getresnet50(num_labels):
    model = models.resnet50(pretrained=True)
    return Resnet50(model, num_labels)