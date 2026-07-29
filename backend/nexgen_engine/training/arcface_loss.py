import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace)
    Paper: https://arxiv.org/abs/1801.07698

    Parameters:
    - in_features: embedding dimension (e.g. 512)
    - out_features: number of identities / classes
    - s: norm feature scale (default 64.0)
    - m: angular margin penalty in radians (default 0.50)
    """
    def __init__(self, in_features=512, out_features=200, s=64.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. Compute sin(theta) and cos(theta + m)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Keep margin valid for large angles
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. One-hot target encoding and apply margin only to ground-truth class
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # 4. Cross Entropy Loss
        loss = F.cross_entropy(output, label)
        return loss
