import torch.nn as nn
import torch
import torchvision.models as models
from models.resnet import resnet50, resnet18
from models.resnet_cifar import ResNet50, ResNet18
from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, model_name, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": resnet18(pretrained=False),
                            "resnet50": resnet50(pretrained=False),
                            "resnet18_cifar": ResNet18(),
                            "resnet50_cifar": ResNet50(),
                           }
        self.num_classes = out_dim
        self.backbone = self.resnet_dict[model_name]
        self.backbone.eval
        self.dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.dim_mlp, self.num_classes)
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(self.dim_mlp, self.dim_mlp), nn.ReLU(), self.backbone.fc)
        # self.backbone.fc = nn.Linear(self.dim_mlp, self.num_classes)

    def forward(self, x):
        return self.backbone(x)

    def remove_projection_head(self):
        self.backbone.fc = nn.Linear(self.dim_mlp, self.num_classes)