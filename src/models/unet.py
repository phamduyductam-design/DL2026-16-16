from torch import nn
from .blocks import Encoder, Up

class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.u3, self.u2, self.u1 = Up(256,128,True), Up(128,64,True), Up(64,32,True)
        self.output = nn.Sequential(nn.Conv2d(32,3,1), nn.Sigmoid())
    def forward(self,x):
        a,b,c,d = self.encoder(x)
        return self.output(self.u1(self.u2(self.u3(d,c),b),a))
