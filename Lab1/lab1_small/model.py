"""Compact complementary CNN for BloodMNIST."""
import torch
import torch.nn as nn

MEAN = [0.79434784, 0.65965901, 0.69619251]
STD = [0.21563033, 0.24160339, 0.11788897]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),
                                 nn.BatchNorm2d(b),nn.ReLU(inplace=True))
        self.features=nn.Sequential(
            block(3,32),block(32,32),nn.MaxPool2d(2),
            block(32,64),block(64,64),nn.MaxPool2d(2),
            block(64,96),block(96,96),nn.AdaptiveAvgPool2d(1))
        self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(96,8))
        self.register_buffer('use_tta',torch.tensor(True))

    def forward_once(self,x):
        return self.head(self.features(x).flatten(1))

    def forward(self,x):
        if self.training or not bool(self.use_tta):
            return self.forward_once(x)
        scores=None
        for flip in (False,True):
            for rotation in range(4):
                view=x.flip(-1) if flip else x
                prob=self.forward_once(view.rot90(rotation,(-2,-1))).softmax(1)
                scores=prob if scores is None else scores+prob
        return (scores/8).clamp_min(1e-8).log()
