import csv
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets import InpaintingDataset
from src.models import make_model
from src.utils.config import ROOT, load_config, project_path, bank_path, experiment_identity
from src.utils.seed import seed_everything, random_states, restore_states, stable_seed
from src.utils.checkpoint import load_checkpoint, commit_training_checkpoints, repair_best_checkpoint
from src.evaluation.metrics import tensor_metrics
from .losses import inpainting_loss, model_input, restore

def get_device(config):
    requested = config.get('device', 'auto')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)

def data_for(config, split, train=False):
    datasets = load_config(config['dataset_config'])
    return InpaintingDataset(datasets['imagenette']['root'],
        project_path(datasets['manifest_dir']) / f'imagenette_{split}.csv', train=train,
        bank=None if train else bank_path(config,'imagenette',split),
        train_seed=config['train_seed'], mask_seed=config['mask_seed'])

def sanity_check(config, model_name, steps=80):
    """Fixed real images and masks; never creates a final experiment checkpoint."""
    seed_everything(config['train_seed'])
    device = get_device(config)
    model = make_model(model_name).to(device)
    dataset = data_for(config, 'train', train=True)
    dataset.train = False
    batch = next(iter(DataLoader(torch.utils.data.Subset(dataset, range(16)), batch_size=16)))
    image, mask = batch['image'].to(device), batch['mask'].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    model.eval()
    with torch.no_grad():
        initial = float(inpainting_loss(model(model_input(image,mask)), image,mask))
    model.train()
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(model_input(image,mask))
        assert prediction.shape == image.shape
        assert prediction.min() >= 0 and prediction.max() <= 1
        loss = inpainting_loss(prediction,image,mask)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('Sanity check: non-finite gradients/loss')
        optimizer.step()
        history.append(float(loss.detach()))
        if step % 10 == 0:
            print(f'Sanity {model_name}: {step}/{steps}, loss={history[-1]:.5f}', flush=True)
    model.eval()
    with torch.no_grad():
        prediction = model(model_input(image,mask))
        final = float(inpainting_loss(prediction,image,mask))
        assert torch.equal(restore(prediction,image,mask)*(1-mask), image*(1-mask))
    if not final < .85*initial:
        raise RuntimeError(f'Overfit did not reduce loss >=15%: {initial:.5f} -> {final:.5f}')
    result = {'model': model_name, 'initial_loss':initial, 'final_loss':final,
              'steps': steps, 'images':16, 'passed':True, 'history':history}
    out = project_path(config['output_dir']) / 'logs'
    out.mkdir(parents=True,exist_ok=True)
    (out / f'{model_name}_sanity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Sanity PASSED: {initial:.5f} -> {final:.5f}',flush=True)
    return result

def train(config, model_name, resume=None):
    config = dict(config, model=model_name)
    identity = experiment_identity(config)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_everything(config['train_seed'])
    device = get_device(config)
    # Full training cannot bypass a real overfit check.
    if resume is None:
        sanity_check(config, model_name)
        seed_everything(config['train_seed'])
    model = make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # PyTorch reduces after num_bad_epochs > patience: 1 means two failed epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=1, threshold=0)
    start, best, best_epoch, bad_epochs = 0, float('inf'), 0, 0
    output = project_path(config['output_dir'])
    run_id = f'{model_name}_seed{config["train_seed"]}'
    best_path = output / 'checkpoints' / f'{run_id}_best.pt'
    best_snapshot = None
    if resume:
        checkpoint = load_checkpoint(resume,device)
        if checkpoint['config'] != config:
            raise ValueError('Resume config differs; use the saved config exactly')
        if checkpoint.get('data_identity') != identity:
            raise ValueError('Resume data changed: manifest/bank fingerprints differ or are missing')
        best_snapshot = repair_best_checkpoint(checkpoint,best_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start, best, best_epoch, bad_epochs = checkpoint['epoch'], checkpoint['best_metric'], checkpoint['best_epoch'], checkpoint['bad_epochs']
        restore_states(checkpoint['random_states'])
    train_data, val_data = data_for(config,'train',True), data_for(config,'val')
    order_generator = torch.Generator()
    loaders = [DataLoader(d,batch_size=config['batch_size'],shuffle=i==0,
                         generator=order_generator if i==0 else torch.Generator().manual_seed(0),
                         num_workers=config['num_workers'], pin_memory=device.type=='cuda')
               for i,d in enumerate([train_data,val_data])]
    log_path = output / 'logs' / f'{run_id}.csv'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    if log_path.exists() and not resume:
        raise FileExistsError(f'Run already exists: {run_id}; resume or use a different output_dir')
    try:
        git_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit = None
    info = {'run_id':run_id,'model':model_name,'train_dataset':'imagenette','train_seed':config['train_seed'],
            'config':config,'git_commit':git_commit,'parameter_count':sum(p.numel() for p in model.parameters()),
            'device':str(device),'GPU':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'torch_version':torch.__version__,'data_identity':identity}
    (log_path.with_suffix('.json')).write_text(json.dumps(info,indent=2),encoding='utf-8')
    fields = ['epoch','epoch_time','train_loss','val_MAE_hole','val_PSNR_hole','lr','best_epoch','checkpoint_path']
    if not log_path.exists():
        with log_path.open('w',newline='') as f:
            csv.DictWriter(f,fieldnames=fields).writeheader()
    elif resume:
        # Remove log rows beyond the committed LAST if a crash left orphan entries.
        with log_path.open(newline='') as f:
            previous_rows = [row for row in csv.DictReader(f) if int(row['epoch']) <= start]
        if checkpoint.get('log_row') and not any(int(row['epoch']) == start for row in previous_rows):
            previous_rows.append(checkpoint['log_row'])
        with log_path.open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader(); writer.writerows(previous_rows)
    best_path = output / 'checkpoints' / f'{run_id}_best.pt'
    for epoch in range(start,config['epochs']):
        if bad_epochs >= config['early_stopping_patience']:
            break
        train_data.epoch = epoch
        order_generator.manual_seed(stable_seed(config['train_seed'],'batch_order',epoch))
        begin, total, n = time.perf_counter(), 0., 0
        model.train()
        for batch_index,batch in enumerate(loaders[0]):
            image,mask = batch['image'].to(device),batch['mask'].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(model_input(image,mask))
            loss = inpainting_loss(prediction,image,mask)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite train loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError('Non-finite training gradient')
            optimizer.step()
            total += float(loss.detach())*len(image)
            n += len(image)
            if (batch_index+1) % 50 == 0:
                print(f'{model_name}: epoch {epoch+1}, batch {batch_index+1}/{len(loaders[0])}, loss={total/n:.5f}',flush=True)
        model.eval()
        mae, psnr = [], []
        with torch.no_grad():
            for batch in loaders[1]:
                image,mask = batch['image'].to(device),batch['mask'].to(device)
                result = tensor_metrics(model(model_input(image,mask)),image,mask)
                mae.extend(result['MAE_hole'].cpu().tolist())
                psnr.extend(result['PSNR_hole'].cpu().tolist())
        metric = float(np.mean(mae))
        scheduler.step(metric)
        improved = metric < best
        if improved:
            best,best_epoch,bad_epochs = metric,epoch+1,0
        else:
            bad_epochs += 1
        best_path = output / 'checkpoints' / f'{run_id}_best.pt'
        payload = {'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                   'scheduler_state_dict':scheduler.state_dict(),'epoch':epoch+1,'best_metric':best,
                   'best_epoch':best_epoch,'bad_epochs':bad_epochs,'random_states':random_states(),
                   'config':config,'run_info':info,'checkpoint_kind':'last','data_identity':identity}
        row = {'epoch':epoch+1,'epoch_time':time.perf_counter()-begin,'train_loss':total/n,
               'val_MAE_hole':metric,'val_PSNR_hole':float(np.mean(psnr)),
               'lr':optimizer.param_groups[0]['lr'],'best_epoch':best_epoch,'checkpoint_path':str(best_path)}
        payload['log_row'] = row
        best_snapshot = commit_training_checkpoints(output / 'checkpoints' / f'{run_id}_last.pt',
                                                    best_path,payload,best_snapshot)
        with log_path.open('a',newline='') as f:
            csv.DictWriter(f,fieldnames=fields).writerow(row)
        print(json.dumps(row),flush=True)
        if bad_epochs >= config['early_stopping_patience']:
            break
    return best_path
