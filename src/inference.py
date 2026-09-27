from functools import lru_cache
import torch
from src.models import make_model
from src.utils.checkpoint import load_checkpoint,model_state_sha256
from src.utils.config import file_sha256,project_path
from src.training.losses import model_input,restore

@lru_cache(maxsize=2)
def _cached_model(checkpoint_path,checkpoint_hash,device):
    checkpoint=load_checkpoint(checkpoint_path,device)
    if checkpoint.get('checkpoint_kind')!='best':
        raise ValueError('Use the best validation checkpoint')
    if model_state_sha256(checkpoint['model_state_dict']) != checkpoint.get('model_state_sha256'):
        raise ValueError('Checkpoint model content hash mismatch')
    model=make_model(checkpoint['config']['model']).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_model(checkpoint_path,device='cpu'):
    path=str(project_path(checkpoint_path))
    return _cached_model(path,file_sha256(path),device)

def predict(image,mask,checkpoint_path,device='cpu'):
    model=load_model(str(checkpoint_path),device)
    with torch.inference_mode():
        prediction=model(model_input(image.to(device),mask.to(device)))
    return prediction.cpu(),restore(prediction.cpu(),image.cpu(),mask.cpu())
