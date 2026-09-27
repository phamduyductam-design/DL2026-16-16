"""Only load checkpoints produced locally by this project (pickle trusted input)."""
import os
import hashlib
import pickle
import torch
from src.utils.config import project_path

def save_checkpoint(path, payload):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)

def load_checkpoint(path, device='cpu'):
    return torch.load(project_path(path), map_location=device, weights_only=False)

def model_state_sha256(state):
    digest = hashlib.sha256()
    for key,tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f'{key}|{value.dtype}|{tuple(value.shape)}'.encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def commit_training_checkpoints(last_path, best_path, payload, previous_best=None):
    """Atomic files plus an embedded BEST snapshot in LAST for crash recovery."""
    if payload['epoch'] == payload['best_epoch']:
        keys = ['config','data_identity','run_info','best_metric','best_epoch']
        best = {key:payload[key] for key in keys}
        best.update(epoch=payload['best_epoch'],checkpoint_kind='best',
                    model_state_dict={k:v.detach().cpu().clone() for k,v in payload['model_state_dict'].items()})
        best['model_state_sha256'] = model_state_sha256(best['model_state_dict'])
    else:
        if previous_best is None:
            raise ValueError('Missing previous BEST snapshot')
        best = previous_best
    if best['best_epoch'] != payload['best_epoch'] or best['best_metric'] != payload['best_metric']:
        raise ValueError('Inconsistent BEST snapshot')
    # If interrupted between saves, old LAST retains its own authoritative BEST.
    save_checkpoint(best_path,best)
    save_checkpoint(last_path,dict(payload,best_checkpoint=best))
    return best

def repair_best_checkpoint(last, best_path):
    """LAST is the transaction commit; repair stale, missing, or newer orphan BEST."""
    best = last.get('best_checkpoint')
    if not best or last.get('checkpoint_kind') != 'last':
        raise ValueError('Resume requires LAST with an embedded BEST snapshot')
    if (best['best_epoch'] != last['best_epoch'] or best['best_metric'] != last['best_metric']
            or best['config'] != last['config'] or best['data_identity'] != last['data_identity']
            or model_state_sha256(best['model_state_dict']) != best['model_state_sha256']):
        raise ValueError('Corrupt LAST/BEST transaction')
    valid = False
    if project_path(best_path).exists():
        try:
            existing = load_checkpoint(best_path)
            valid = (existing.get('checkpoint_kind') == 'best'
                and existing.get('best_epoch') == best['best_epoch']
                and existing.get('best_metric') == best['best_metric']
                and existing.get('data_identity') == best['data_identity']
                and existing.get('config') == best['config']
                and model_state_sha256(existing['model_state_dict']) == best['model_state_sha256'])
        except (OSError,RuntimeError,EOFError,KeyError,ValueError,pickle.UnpicklingError):
            valid = False
    if not valid:
        save_checkpoint(best_path,best)
    return best
