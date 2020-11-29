import numpy as np
import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES']='2'
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from MobileNetV2 import MobileNetV2
import pdb


valdir="E:/coco/val2017/000000000285.jpg"

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])))

print(val_loader)