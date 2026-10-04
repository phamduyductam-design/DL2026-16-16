import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import torch
import gradio as gr
from src.datasets import preprocess
from src.masks import generate_mask
from src.inference import predict
from src.evaluation.metrics import metrics
from src.utils.config import project_path

def create_app(checkpoint_dir='outputs/checkpoints',seed=42):
    def generate(image,pattern,ratio,mask_seed):
        if image is None: raise gr.Error('Upload an image first')
        tensor=preprocess(image).unsqueeze(0)
        mask,meta=generate_mask(pattern,float(ratio)/100,int(mask_seed))
        mask=torch.from_numpy(mask.copy()).float()[None,None]
        state={'image':tensor,'mask':mask,'actual_ratio':meta['actual_ratio']}
        to_array=lambda t:t[0].numpy().transpose(1,2,0)
        return state,to_array(tensor),mask[0,0].numpy(),to_array(tensor*(1-mask)),f'Actual area: {meta["actual_ratio"]:.2%}'
    def run(state,model):
        if state is None: raise gr.Error('Generate a mask first')
        path=project_path(checkpoint_dir)/f'{model}_seed{seed}_best.pt'
        if not path.exists(): raise gr.Error(f'Checkpoint missing: {path}. Train this model first.')
        prediction,restored=predict(state['image'],state['mask'],path,'cuda' if torch.cuda.is_available() else 'cpu')
        result=metrics(prediction,state['image'],state['mask'])
        text='Synthetic-mask reconstruction metrics against the uploaded reference (not confidence):\n'
        text+=f'Actual area: {state["actual_ratio"]:.2%}\n'+ '\n'.join(f'{k}: {float(v[0]):.5f}' for k,v in result.items())
        return restored[0].numpy().transpose(1,2,0),text
    with gr.Blocks(title='Deep Image Inpainting') as app:
        gr.Markdown('# Deep Image Inpainting\nGenerate one fixed mask, then compare both models.')
        gr.Markdown('The uploaded image is treated as the reference and is then masked synthetically. '
                    'MAE, PSNR, and SSIM measure similarity to that reference; they are **not confidence scores**. '
                    'The demo cannot verify recovery of an already damaged image without its original. '
                    'Switching models keeps the same uploaded image and generated mask.')
        state=gr.State(None)
        upload=gr.Image(type='pil',label='Upload image')
        with gr.Row():
            model=gr.Dropdown(['autoencoder','unet'],value='autoencoder',label='Model')
            pattern=gr.Dropdown([('Center Rectangle','C'),('Random Rectangle','R'),('Free-form','F'),('Multiple Holes','H')],value='C',label='Pattern')
            ratio=gr.Dropdown([15,30,50],value=30,label='Missing area (%)')
            mask_seed=gr.Number(value=42,precision=0,label='Mask seed')
        with gr.Row():
            make=gr.Button('Generate mask')
            infer=gr.Button('Run model')
        with gr.Row():
            truth=gr.Image(label='Ground Truth')
            mask_view=gr.Image(label='Mask')
            masked=gr.Image(label='Masked Image')
            restored=gr.Image(label='Restored Image')
        scores=gr.Textbox(label='Metrics')
        make.click(generate,[upload,pattern,ratio,mask_seed],[state,truth,mask_view,masked,scores])
        infer.click(run,[state,model],[restored,scores])
        model.change(run,[state,model],[restored,scores])
    return app

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir',default='outputs/checkpoints')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    create_app(args.checkpoint_dir,args.seed).launch(server_name='127.0.0.1',share=False)
