import torch
from torch import nn
from torch.nn import functional as F

def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())

class Up(nn.Module):
    def __init__(self, cin, cout, skip=False):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout, 3, padding=1)
        self.block = block(cout * (2 if skip else 1), cout)
    def forward(self, x, skip=None):
        x = self.reduce(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        if skip is not None:
            if x.shape != skip.shape:
                raise ValueError(f'Skip mismatch: {x.shape}, {skip.shape}')
            x = torch.cat([x, skip], dim=1)
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3, self.b = block(4,32), block(32,64), block(64,128), block(128,256)
        self.pool = nn.MaxPool2d(2)
    def forward(self,x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.b(self.pool(c))
        return a,b,c,d
