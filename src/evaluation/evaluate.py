import json
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.models import make_model
from src.datasets import InpaintingDataset
from src.utils.config import load_config, project_path, bank_path, experiment_identity, file_sha256
from src.utils.checkpoint import load_checkpoint, model_state_sha256
from src.training.engine import get_device
from src.training.losses import model_input
from .metrics import metrics

def evaluate(checkpoint_path, dataset, dataset_config=None, batch_size=32):
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get('checkpoint_kind') != 'best':
        raise ValueError('Final evaluation requires an Imagenette-validation BEST checkpoint')
    config = checkpoint['config']
    dataset_config = config['dataset_config'] if dataset_config is None else dataset_config
    live_config=dict(config,dataset_config=dataset_config)
    identity=experiment_identity(live_config)
    if checkpoint.get('data_identity') != identity:
        raise ValueError('Evaluation manifests/banks differ from the checkpoint experiment')
    if model_state_sha256(checkpoint['model_state_dict']) != checkpoint.get('model_state_sha256'):
        raise ValueError('BEST model content hash mismatch')
    checkpoint_hash=file_sha256(checkpoint_path)
    name,seed = config['model'],config['train_seed']
    device = get_device(config)
    model = make_model(name).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    d=load_config(dataset_config)
    data = InpaintingDataset(d[dataset]['root'], project_path(d['manifest_dir'])/f'{dataset}_test.csv',
                             bank=bank_path(config,dataset,'test'))
    loader = DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=0)
    rows=[]
    with torch.inference_mode():
        # Warm-up is excluded from timing; measured duration is model-only throughput/image.
        first=next(iter(loader))
        model(model_input(first['image'].to(device),first['mask'].to(device)))
        for batch in loader:
            image,mask=batch['image'].to(device),batch['mask'].to(device)
            inputs=model_input(image,mask)
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            prediction=model(inputs)
            if device.type=='cuda': torch.cuda.synchronize()
            duration=(time.perf_counter()-start)/len(image)*1000
            result=metrics(prediction,image,mask)
            for i in range(len(image)):
                condition=batch['condition'][i]
                rows.append({'train_dataset':'imagenette','eval_dataset':dataset,'model':name,'train_seed':seed,
                    'image_id':batch['image_id'][i],'mask_id':batch['mask_id'][i],
                    'pattern':condition[0],'size':int(condition[1:]),'condition':condition,
                    'actual_ratio':float(batch['actual_ratio'][i]),
                    **{k:float(v[i]) for k,v in result.items()},'inference_ms':duration,
                    'parameter_count':sum(p.numel() for p in model.parameters()),'timing_batch_size':batch_size,
                    'checkpoint_sha256':checkpoint_hash,'manifest_sha256':identity['manifests'][f'{dataset}_test'],
                    'bank_sha256':identity['banks'][f'{dataset}_test']['bank_sha256']})
            if len(rows)%1200==0: print(f'{name}/{dataset}: {len(rows)}/{len(data)}',flush=True)
    out=project_path(config['output_dir'])/'metrics'
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    expected=12000 if dataset=='imagenette' else 6000
    if len(frame)!=expected or frame['mask_id'].nunique()!=expected:
        raise ValueError(f'Wrong pair count: {len(frame)}, expected {expected}')
    per=out/f'{name}_seed{seed}_{dataset}_per_image.csv'
    frame.to_csv(per,index=False)
    summary=aggregate(frame)
    summary.to_csv(out/f'{name}_seed{seed}_{dataset}_summary.csv',index=False)
    print(f'Saved {per}',flush=True)
    return frame

def aggregate(frame):
    keys=['train_dataset','eval_dataset','model','train_seed','pattern','size','condition']
    values=['actual_ratio','MAE_hole','MSE_hole','PSNR_hole','PSNR_full','SSIM_full','inference_ms','parameter_count']
    grouped=frame.groupby(keys,sort=True)
    result=grouped[values].mean().reset_index()
    result['n_pairs']=grouped.size().to_numpy()
    return result

def combine_results(output='outputs',seed=42):
    out=project_path(output)/'metrics'
    files=[out/f'{m}_seed{seed}_{d}_per_image.csv' for m in ['autoencoder','unet'] for d in ['imagenette','pets']]
    if not all(p.exists() for p in files):
        raise FileNotFoundError('Need all four model/dataset evaluations before combining')
    frames=[pd.read_csv(p) for p in files]
    a,b=frames[:2],frames[2:]
    for left,right in zip(a,b):
        paired=['image_id','mask_id','condition','actual_ratio','manifest_sha256','bank_sha256','timing_batch_size']
        if not left[paired].equals(right[paired]):
            raise ValueError('Models were evaluated on different image/mask pairs')
    for first in frames:
        # Per-image pairing is stricter than a dataset-level area comparison.
        areas=first.groupby(['image_id','size']).actual_ratio.agg(['min','max'])
        if ((areas['max']-areas['min']) > .01).any():
            raise ValueError('Evaluation patterns have unmatched areas')
    for model_frames in [frames[:2],frames[2:]]:
        hashes={h for f in model_frames for h in f.checkpoint_sha256.unique()}
        if len(hashes)!=1:
            raise ValueError('Domain evaluations used different checkpoints')
    frame=pd.concat(frames,ignore_index=True)
    summary=aggregate(frame)
    if len(summary)!=48 or len(frame)!=36000:
        raise ValueError('Expected 48 summary rows and 36000 pairs per seed')
    frame.to_csv(out/'per_image.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    frame.to_csv(out/f'per_image_seed{seed}.csv',index=False)
    summary.to_csv(out/f'summary_seed{seed}.csv',index=False)
    print('Validated 48 rows/seed and 36000 paired evaluations')
    return summary
