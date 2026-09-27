import _bootstrap
import argparse
from src.utils.config import load_config
from src.training.engine import train,sanity_check

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['autoencoder','unet'],required=True)
    p.add_argument('--config',default='configs/base.yaml')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume')
    p.add_argument('--sanity-only',action='store_true')
    args=p.parse_args()
    c=load_config(args.config)
    c['train_seed']=args.seed
    (sanity_check(c,args.model) if args.sanity_only else train(c,args.model,args.resume))
if __name__=='__main__': main()
