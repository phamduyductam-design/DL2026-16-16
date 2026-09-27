import _bootstrap
import argparse
from src.utils.config import load_config
from src.datasets import prepare_manifests

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/datasets.yaml')
    args=p.parse_args()
    prepare_manifests(load_config(args.config))
if __name__=='__main__': main()
