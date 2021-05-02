from torchvision import models
import torch.nn as nn
import torch
import os 
def ResNet50(num_classes = 196):
    model_resnet = models.resnet50(pretrained=True)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def ResNet152(num_classes = 196):
    model_resnet = models.resnet152(pretrained=True)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)
    return model_resnet