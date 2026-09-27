"""Sequential validated pipeline. Re-running preserves manifests/banks and resumes LAST."""
import _bootstrap
import argparse
import subprocess
import sys
from src.utils.config import project_path,load_config

def run(*args):
    subprocess.run([sys.executable,*args],cwd=project_path('.'),check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--config',default='configs/base.yaml')
    a=p.parse_args()
    c=load_config(a.config)
    d=load_config(c['dataset_config'])
    output=project_path(c['output_dir'])
    manifests=project_path(d['manifest_dir'])
    names=['imagenette_train','imagenette_val','imagenette_test','pets_test']
    if not any((manifests/f'{n}.csv').exists() for n in names):
        run('scripts/prepare_data.py','--config',c['dataset_config'])
    run('-m','pytest','-q')
    run('scripts/build_mask_bank.py','--config',a.config)
    for model in ['autoencoder','unet']:
        last=output/'checkpoints'/f'{model}_seed{a.seed}_last.pt'
        command=['scripts/train.py','--model',model,'--config',a.config,'--seed',str(a.seed)]
        if last.exists(): command += ['--resume',str(last)]
        run(*command)
        for dataset in ['imagenette','pets']:
            run('scripts/evaluate.py','--checkpoint',str(output/'checkpoints'/f'{model}_seed{a.seed}_best.pt'),
                '--dataset',dataset,'--dataset-config',c['dataset_config'],'--batch-size',str(c['batch_size']))
    run('scripts/evaluate.py','--combine','--seed',str(a.seed),'--output',str(output))
    run('scripts/plot_results.py','--output',str(output),'--dataset-config',c['dataset_config'],
        '--mask-bank-dir',c.get('mask_bank_dir','data/mask_bank'))
    run('scripts/check_project.py','--seed',str(a.seed),'--config',a.config)
if __name__=='__main__': main()
