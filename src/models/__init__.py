from .autoencoder import Autoencoder
from .unet import SmallUNet

def make_model(name):
    if name == 'autoencoder':
        return Autoencoder()
    if name == 'unet':
        return SmallUNet()
    raise ValueError(f'Unknown model: {name}')
