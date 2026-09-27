import _bootstrap
import argparse
from src.evaluation.evaluate import evaluate,combine_results
from src.utils.checkpoint import load_checkpoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['autoencoder','unet'])
    p.add_argument('--checkpoint')
    p.add_argument('--dataset',choices=['imagenette','pets'],default='imagenette')
    p.add_argument('--dataset-config')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--combine',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default='outputs')
    args=p.parse_args()
    if args.combine:
        combine_results(args.output,args.seed)
    else:
        if not args.checkpoint: p.error('--checkpoint is required')
        c=load_checkpoint(args.checkpoint)
        if args.model and c['config']['model']!=args.model: p.error('Checkpoint model mismatch')
        evaluate(args.checkpoint,args.dataset,args.dataset_config,args.batch_size)
if __name__=='__main__': main()
