import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class PretrainedFeatureExtractor(nn.Module):
    def __init__(self, out_channels=384):
        super().__init__()
        self.pretrained_model = models.wide_resnet101_2(weights="Wide_ResNet101_2_Weights.IMAGENET1K_V1")
        self.out_channels = out_channels

    def forward(self, x):
        x = self.pretrained_model.conv1(x)
        x = self.pretrained_model.bn1(x)
        x = self.pretrained_model.relu(x)
        x = self.pretrained_model.maxpool(x)
        x = self.pretrained_model.layer1(x)
        pretrained_output1 = self.pretrained_model.layer2(x)
        pretrained_output2 = self.pretrained_model.layer3(pretrained_output1)

        b, c, h, w = pretrained_output1.shape
        pretrained_output2 = F.interpolate(pretrained_output2, size=(h, w), mode="bilinear", align_corners=False)
        features = torch.cat([pretrained_output1, pretrained_output2], dim=1)
        b, c, h, w = features.shape
        features = features.reshape(b, c, h * w)
        features = features.transpose(1, 2)
        target_features = F.adaptive_avg_pool1d(features, self.out_channels)
        target_features = target_features.transpose(1, 2)
        target_features = target_features.reshape(b, self.out_channels, h, w)
        return target_features
