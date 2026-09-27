import numpy as np
import torch
from skimage.metrics import structural_similarity
from src.training.losses import region_mean, restore

def psnr(mse):
    return torch.where(mse == 0, torch.full_like(mse,float('inf')), -10*torch.log10(mse))

def tensor_metrics(prediction,image,mask):
    error = prediction-image
    mae, mse = region_mean(error.abs(),mask), region_mean(error.square(),mask)
    full_mse = (restore(prediction,image,mask)-image).square().mean(dim=(1,2,3))
    return {'MAE_hole': mae, 'MSE_hole': mse, 'PSNR_hole': psnr(mse), 'PSNR_full': psnr(full_mse)}

def metrics(prediction,image,mask):
    result = {k:v.detach().cpu().numpy() for k,v in tensor_metrics(prediction,image,mask).items()}
    restored = restore(prediction,image,mask).detach().cpu().numpy().transpose(0,2,3,1)
    truth = image.detach().cpu().numpy().transpose(0,2,3,1)
    result['SSIM_full'] = np.asarray([structural_similarity(a,b,data_range=1.,channel_axis=-1)
                                     for a,b in zip(restored,truth)])
    return result
