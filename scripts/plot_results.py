import _bootstrap
import argparse
from src.evaluation.plots import mask_figures,training_figures,analysis_figures

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='outputs')
    p.add_argument('--dataset-config',default='configs/datasets.yaml')
    p.add_argument('--masks-only',action='store_true')
    p.add_argument('--mask-bank-dir',default='data/mask_bank')
    a=p.parse_args()
    mask_figures(a.output,a.mask_bank_dir)
    if not a.masks_only:
        training_figures(a.output)
        analysis_figures(a.output,a.dataset_config,a.mask_bank_dir)
