import _bootstrap
import argparse
from src.utils.config import load_config,project_path,bank_path
from src.datasets.imagenette import validate_manifests
from src.masks.bank import build_bank, validate_generator

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/base.yaml')
    args=p.parse_args()
    c=load_config(args.config)
    d=load_config(c['dataset_config'])
    validate_manifests(d)
    validate_generator(100)
    for dataset,split in [('imagenette','val'),('imagenette','test'),('pets','test')]:
        out=bank_path(c,dataset,split)
        build_bank(project_path(d['manifest_dir'])/f'{dataset}_{split}.csv',out,dataset,split,c['mask_seed'])
    from src.evaluation.plots import mask_figures
    mask_figures(c['output_dir'],c.get('mask_bank_dir','data/mask_bank'))
if __name__=='__main__': main()
