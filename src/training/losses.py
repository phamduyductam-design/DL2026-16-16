import torch

def region_mean(error, mask):
    return (error*mask).sum(dim=(1,2,3)) / (3*mask.sum(dim=(1,2,3))).clamp_min(1e-8)

def inpainting_loss(prediction, image, mask):
    error = (prediction-image).abs()
    return (region_mean(error, mask) + .1*region_mean(error,1-mask)).mean()

def model_input(image,mask):
    return torch.cat([image*(1-mask),mask],dim=1)

def restore(prediction,image,mask):
    return prediction*mask + image*(1-mask)
