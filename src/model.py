import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology
from torchvision import transforms
from torchvision import models


class kl_loss(nn.Module):
    def __init__(self, w1, w2, w3, w4, w5):
        super().__init__()
        self.vgg = vgg16_feature().cuda()
        self.criterion = nn.KLDivLoss()


    def forward(self, input, gt):
        input_vgg = self.vgg(input)
        gt_vgg = self.vgg(gt)

        map_A = similarity_map(input_vgg)
        map_B = similarity_map(gt_vgg)
        loss = self.criterion(map_A, map_B.detach())

        return loss


