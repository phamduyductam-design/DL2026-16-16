"""Figures/tables use measured outputs only. Matplotlib uses a noninteractive backend."""
import csv
import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.utils.config import project_path, load_config
from src.masks.generators import CONDITIONS,generate_mask

def save(fig,path):
    path=project_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not getattr(fig, '_inpainting_manual_layout', False):
        fig.tight_layout()
    fig.savefig(path,dpi=220,bbox_inches='tight')
    plt.close(fig)

def mask_figures(output='outputs',mask_bank_dir='data/mask_bank'):
    out=project_path(output)/'figures'
    fig,axes=plt.subplots(4,3,figsize=(8,10))
    for ax,condition in zip(axes.flat,CONDITIONS):
        mask,meta=generate_mask(condition[0],int(condition[1:])/100,42)
        ax.imshow(mask,cmap='gray',vmin=0,vmax=1)
        ax.set_title(f'{condition}: actual {meta["actual_ratio"]:.1%}')
        ax.set_xlabel('x (pixels)'); ax.set_ylabel('y (pixels)')
    fig.suptitle('12 missing-region conditions: white = missing')
    save(fig,out/'mask_examples.png')
    metadata=[]
    for file in project_path(mask_bank_dir).rglob('metadata.csv'):
        metadata.append(pd.read_csv(file))
    if metadata:
        frame=pd.concat(metadata)
        fig,axes=plt.subplots(4,3,figsize=(12,10))
        for ax,condition in zip(axes.flat,CONDITIONS):
            values=frame.loc[frame.condition==condition,'actual_ratio']*100
            ax.hist(values,bins=20,label=condition)
            ax.axvline(int(condition[1:]),color='red',linestyle='--',label='Target')
            ax.set_title(condition); ax.set_xlabel('Actual missing area (%)'); ax.set_ylabel('Count'); ax.legend()
        fig.suptitle('Fixed mask banks: actual area distribution')
        save(fig,out/'mask_area_histogram.png')

def training_figures(output='outputs'):
    out=project_path(output)
    for path in (out/'logs').glob('*_seed*.csv'):
        frame=pd.read_csv(path)
        if frame.empty: continue
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].plot(frame.epoch,frame.train_loss,label='Train total loss')
        axes[0].plot(frame.epoch,frame.val_MAE_hole,label='Validation MAE-hole')
        axes[0].set_ylabel('Loss / MAE'); axes[0].legend()
        axes[1].plot(frame.epoch,frame.val_PSNR_hole,label='Validation PSNR-hole')
        axes[1].set_ylabel('PSNR-hole (dB)'); axes[1].legend()
        for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
        fig.suptitle(path.stem)
        save(fig,out/'figures'/f'{path.stem}_training.png')

def analysis_figures(output='outputs',dataset_config='configs/datasets.yaml',mask_bank_dir='data/mask_bank'):
    out=project_path(output)
    summary=pd.read_csv(out/'metrics/summary.csv')
    per=pd.read_csv(out/'metrics/per_image.csv')
    if len(summary)!=48: raise ValueError('Need a complete single-seed 48-row summary')
    figures=out/'figures'
    for dataset in ['imagenette','pets']:
        frame=summary[summary.eval_dataset==dataset]
        # E1: condition-level comparisons, rather than collapsing dataset scores.
        for metric in ['MAE_hole','PSNR_hole','SSIM_full','inference_ms']:
            pivot=frame.pivot(index='condition',columns='model',values=metric).reindex(CONDITIONS)
            ax=pivot.plot.bar(figsize=(12,4),rot=0)
            ax.set_title(f'E1 {dataset}: model comparison'); ax.set_xlabel('Condition'); ax.set_ylabel(metric)
            save(ax.figure,figures/f'{dataset}_model_{metric}.png')
        # E2: hold model and pattern fixed.
        for metric in ['MAE_hole','PSNR_hole']:
            fig,axes=plt.subplots(1,4,figsize=(15,4))
            for ax,pattern in zip(axes,'CRFH'):
                for model in ['autoencoder','unet']:
                    rows=frame[(frame.pattern==pattern)&(frame.model==model)].sort_values('size')
                    ax.plot(rows['size'],rows[metric],marker='o',label=model)
                ax.set_title(f'{dataset}: {pattern}'); ax.set_xlabel('Target missing area (%)'); ax.set_ylabel(metric); ax.legend()
            fig.suptitle(f'E2 Mask size: {metric}')
            save(fig,figures/f'{dataset}_size_{metric}.png')
        # E3: comparable target area.
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,size in zip(axes,[15,30,50]):
            rows=frame[(frame['size']==size)&frame.pattern.isin(['R','F','H'])]
            rows.pivot(index='pattern',columns='model',values='MAE_hole').reindex(['R','F','H']).plot.bar(ax=ax,rot=0)
            ax.set_title(f'{dataset} {size}%'); ax.set_xlabel('Pattern'); ax.set_ylabel('MAE-hole')
        fig.suptitle('E3 Pattern comparison at equal target area')
        save(fig,figures/f'{dataset}_pattern.png')
        # E4: paired masks share width/height.
        rows=frame[frame.pattern.isin(['C','R'])]
        ax=rows.pivot(index=['size','pattern'],columns='model',values='MAE_hole').plot.bar(figsize=(10,4),rot=0)
        ax.set_title(f'E4 {dataset}: center vs random position'); ax.set_xlabel('(Area %, pattern)'); ax.set_ylabel('MAE-hole')
        save(ax.figure,figures/f'{dataset}_position.png')
        pairs=per[(per.eval_dataset==dataset)&per.pattern.isin(['C','R'])]
        paired=pairs.pivot(index=['model','image_id','size'],columns='pattern',values='MAE_hole')
        paired['random_minus_center']=paired['R']-paired['C']
        paired.reset_index().to_csv(out/'metrics'/f'{dataset}_paired_position.csv',index=False)
        # Heatmap includes all twelve named conditions.
        matrix=frame.pivot(index='model',columns='condition',values='MAE_hole').reindex(columns=CONDITIONS)
        fig,ax=plt.subplots(figsize=(12,3))
        im=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='viridis')
        ax.set_xticks(range(12),CONDITIONS); ax.set_yticks(range(2),matrix.index)
        ax.set_xlabel('Condition'); ax.set_ylabel('Model'); ax.set_title(f'{dataset}: MAE-hole heatmap')
        fig.colorbar(im,ax=ax,label='MAE-hole')
        save(fig,figures/f'{dataset}_heatmap.png')
    # E6: compare domains condition by condition with the same checkpoint.
    for model in ['autoencoder','unet']:
        frame=summary[summary.model==model]
        ax=frame.pivot(index='condition',columns='eval_dataset',values='MAE_hole').reindex(CONDITIONS).plot.bar(figsize=(12,4),rot=0)
        ax.set_title(f'E6 {model}: Imagenette vs Oxford-IIIT Pet'); ax.set_xlabel('Condition'); ax.set_ylabel('MAE-hole')
        save(ax.figure,figures/f'{model}_domain_comparison.png')
    summary.to_csv(out/'metrics/E1_E6_table.csv',index=False)
    qualitative_figures(output,dataset_config,mask_bank_dir)
    report_tables(output)

def qualitative_figures(output,dataset_config,mask_bank_dir='data/mask_bank'):
    from src.datasets import preprocess
    from src.datasets.imagenette import read_bytes
    from src.inference import predict
    out=project_path(output)
    per=pd.read_csv(out/'metrics/per_image.csv')
    d=load_config(dataset_config)
    seed=int(per.train_seed.iloc[0])
    selections=[]
    for dataset in ['imagenette','pets']:
        manifest=pd.read_csv(project_path(d['manifest_dir'])/f'{dataset}_test.csv',index_col='image_id')
        bank=project_path(mask_bank_dir)/dataset/'test'
        masks=pd.read_csv(bank/'metadata.csv',index_col='mask_id')
        for condition in CONDITIONS:
            rows=per[(per.eval_dataset==dataset)&(per.condition==condition)]
            ranking=rows.groupby(['image_id','mask_id']).MAE_hole.mean().reset_index().sort_values(['MAE_hole','image_id'])
            indices=[0,(len(ranking)-1)//2,len(ranking)-1]
            fig,axes=plt.subplots(3,6,figsize=(15,8))
            for axrow,rank,label in zip(axes,indices,['good','median','bad']):
                row=ranking.iloc[rank]
                selections.append({'eval_dataset':dataset,'condition':condition,'rank_rule':'paired mean MAE-hole; tie=image_id',
                                   'rank':rank,'kind':label,'image_id':row.image_id,'mask_id':row.mask_id,'MAE_hole':row.MAE_hole})
                with Image.open(io.BytesIO(read_bytes(d[dataset]['root'],manifest.loc[row.image_id,'path']))) as image:
                    image=preprocess(image)[None]
                with Image.open(bank/masks.loc[row.mask_id,'mask_path']) as im:
                    mask=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255)[None,None]
                outputs=[]
                for model in ['autoencoder','unet']:
                    _,restored=predict(image,mask,out/'checkpoints'/f'{model}_seed{seed}_best.pt',
                                       'cuda' if torch.cuda.is_available() else 'cpu')
                    outputs.append(restored)
                views=[image,image*(1-mask),*outputs]
                for ax,tensor,title in zip(axrow[:4],views,['Ground Truth','Masked','Autoencoder','U-Net']):
                    ax.imshow(tensor[0].numpy().transpose(1,2,0)); ax.set_title(f'{label}: {title}'); ax.axis('off')
                for ax,tensor,title in zip(axrow[4:],outputs,['AE error','U-Net error']):
                    error=(tensor-image).abs().mean(1)[0].numpy()
                    im=ax.imshow(error,vmin=0,vmax=1,cmap='magma'); ax.set_title(title); ax.axis('off')
                fig.colorbar(im,ax=list(axrow[4:]),fraction=.025,pad=.01,label='RGB mean absolute error [0,1]')
            fig.suptitle(f'E5 {dataset} {condition}: fixed good / median / bad ranking')
            fig.subplots_adjust(top=.90,wspace=.12,hspace=.25,right=.94)
            fig._inpainting_manual_layout = True
            save(fig,out/'figures'/f'{dataset}_{condition}_failure_grid.png')
    pd.DataFrame(selections).to_csv(out/'metrics/qualitative_selections.csv',index=False)

def report_tables(output='outputs'):
    summary=pd.read_csv(project_path(output)/'metrics/summary.csv')
    lines=['# Measured results','', 'Generated from summary.csv; datasets and conditions remain separate.','',
           '| Dataset | Model | Condition | MAE-hole | PSNR-hole (dB) | SSIM-full | Inference ms/image |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.eval_dataset} | {r.model} | {r.condition} | {r.MAE_hole:.5f} | {r.PSNR_hole:.3f} | {r.SSIM_full:.4f} | {r.inference_ms:.3f} |')
    project_path('report/results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Keep presentation prose tied to measured values, separately for each domain.
    slides=project_path('report/slides.md').read_text(encoding='utf-8')
    slides=slides.replace('To be measured.', 'Measured figures and tables: see report/results.md.')
    slides=slides.replace('Pending experiment.', 'Use the measured condition-wise results in report/results.md.')
    measured=['','## Measured model comparison (equal weight over 12 conditions, separately by dataset)','',
              '| Dataset | Model | MAE-hole | PSNR-hole | SSIM-full | Parameters | ms/image |',
              '|---|---|---:|---:|---:|---:|---:|']
    for (dataset,model),rows in summary.groupby(['eval_dataset','model']):
        measured.append(f'| {dataset} | {model} | {rows.MAE_hole.mean():.5f} | {rows.PSNR_hole.mean():.3f} | {rows.SSIM_full.mean():.4f} | {int(rows.parameter_count.iloc[0])} | {rows.inference_ms.mean():.3f} |')
    # Idempotent regeneration rather than appending duplicate tables.
    slides=slides.split('\n## Measured model comparison')[0]
    project_path('report/slides.md').write_text(slides+'\n'.join(measured)+'\n',encoding='utf-8')
    outline=project_path('report/outline.md').read_text(encoding='utf-8')
    outline=outline.replace('Pending experiment.', 'Measured outputs available: see [results.md](results.md) and outputs/figures/.')
    project_path('report/outline.md').write_text(outline,encoding='utf-8')
