from torch import nn
from .blocks import Encoder, Up

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.u3, self.u2, self.u1 = Up(256,128), Up(128,64), Up(64,32)
        self.output = nn.Sequential(nn.Conv2d(32,3,1), nn.Sigmoid())
    def forward(self,x):
        _,_,_,x = self.encoder(x)
        return self.output(self.u1(self.u2(self.u3(x))))
