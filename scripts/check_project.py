import _bootstrap
import argparse
import json
import pandas as pd
import numpy as np
from src.utils.config import project_path,load_config,bank_path,file_sha256
from src.utils.checkpoint import load_checkpoint,model_state_sha256
from src.datasets.imagenette import validate_manifests
from src.masks.bank import validate_bank
from src.masks.generators import CONDITIONS

def check(seed=42,config_path='configs/base.yaml',write_status=True):
    c=load_config(config_path)
    c['train_seed']=seed
    d=load_config(c['dataset_config'])
    out=project_path(c['output_dir'])
    manifest_dir=project_path(d['manifest_dir'])
    status,errors={},{}
    def verify(name,operation):
        try:
            if operation() is False:
                raise ValueError('Required artifact missing/inconsistent')
            status[name]=True
        except (OSError,ValueError,KeyError,RuntimeError,EOFError,AssertionError,pd.errors.ParserError) as error:
            status[name]=False
            errors[name]=str(error)
    verify('locked_manifests',lambda:validate_manifests(d))
    banks={}
    for dataset,split in [('imagenette','val'),('imagenette','test'),('pets','test')]:
        def operation(dataset=dataset,split=split):
            banks[f'{dataset}_{split}']=validate_bank(manifest_dir/f'{dataset}_{split}.csv',
                bank_path(c,dataset,split),dataset,split,c['mask_seed'])
        verify(f'bank_{dataset}_{split}',operation)
    identity=None
    if status['locked_manifests'] and len(banks)==3:
        identity={'manifests':{n:file_sha256(manifest_dir/f'{n}.csv') for n in
                  ['imagenette_train','imagenette_val','imagenette_test','pets_test']},
                  'banks':banks,'audit_sha256':file_sha256(manifest_dir/'audit.json')}
    checkpoint_hashes={}
    for model in ['autoencoder','unet']:
        def verify_run(model=model):
            run_id=f'{model}_seed{seed}'
            path=out/'checkpoints'/f'{run_id}_best.pt'
            best=load_checkpoint(path)
            last=load_checkpoint(out/'checkpoints'/f'{run_id}_last.pt')
            if identity is None or best.get('data_identity')!=identity or last.get('data_identity')!=identity:
                raise ValueError('Checkpoint data fingerprint mismatch')
            if best.get('config')!=dict(c,model=model) or last.get('config')!=dict(c,model=model):
                raise ValueError('Run configuration mismatch')
            if best.get('checkpoint_kind')!='best' or last.get('checkpoint_kind')!='last':
                raise ValueError('Wrong checkpoint types')
            embedded=last['best_checkpoint']
            if (best['best_epoch']!=last['best_epoch'] or best['best_metric']!=last['best_metric']
                    or model_state_sha256(best['model_state_dict'])!=best['model_state_sha256']
                    or model_state_sha256(embedded['model_state_dict'])!=best['model_state_sha256']):
                raise ValueError('BEST/LAST are inconsistent')
            if last['epoch']<c['epochs'] and last['bad_epochs']<c['early_stopping_patience']:
                raise ValueError('Training has not completed')
            logs=pd.read_csv(out/'logs'/f'{run_id}.csv')
            if logs.empty or int(logs.epoch.iloc[-1])!=last['epoch']:
                raise ValueError('Incomplete training logs')
            if abs(logs.val_MAE_hole.min()-best['best_metric'])>1e-10:
                raise ValueError('BEST does not match validation logs')
            sanity=json.loads((out/'logs'/f'{model}_sanity.json').read_text())
            if sanity.get('passed') is not True or sanity['final_loss']>=.85*sanity['initial_loss']:
                raise ValueError('Missing/failed real sanity check')
            checkpoint_hashes[model]=file_sha256(path)
        verify(f'{model}_completed_run',verify_run)
    def verify_results():
        summary=pd.read_csv(out/'metrics/summary.csv')
        per=pd.read_csv(out/'metrics/per_image.csv')
        expected={(m,dataset,condition) for m in ['autoencoder','unet'] for dataset in ['imagenette','pets'] for condition in CONDITIONS}
        actual=set(zip(summary.model,summary.eval_dataset,summary.condition))
        if (len(summary)!=48 or actual!=expected or len(per)!=36000 or not (per.train_seed==seed).all()
                or not (summary.train_seed==seed).all() or not (per.train_dataset=='imagenette').all()
                or not (summary.train_dataset=='imagenette').all()):
            raise ValueError('Expected complete 48-row/36000-pair results for this seed')
        for metric in ['MAE_hole','MSE_hole','actual_ratio']:
            if not np.isfinite(per[metric]).all() or not per[metric].between(0,1).all():
                raise ValueError(f'Invalid measured {metric}')
        if not np.isfinite(per.SSIM_full).all() or not per.SSIM_full.between(-1,1).all():
            raise ValueError('Invalid SSIM-full')
        if not np.isfinite(per.inference_ms).all() or (per.inference_ms<=0).any():
            raise ValueError('Invalid inference timing')
        for metric in ['PSNR_hole','PSNR_full']:
            if per[metric].isna().any() or (per[metric]<0).any():
                raise ValueError(f'Invalid measured {metric}')
        if identity is None or len(checkpoint_hashes)!=2:
            raise ValueError('Cannot certify results without valid complete runs')
        for (model,dataset),rows in per.groupby(['model','eval_dataset']):
            manifest=pd.read_csv(manifest_dir/f'{dataset}_test.csv')
            expected_pairs={(i,condition) for i in manifest.image_id for condition in CONDITIONS}
            if len(rows)!=len(expected_pairs) or set(zip(rows.image_id,rows.condition))!=expected_pairs:
                raise ValueError('Wrong per-image evaluation pairs')
            if not (rows.checkpoint_sha256==checkpoint_hashes[model]).all():
                raise ValueError('Stale evaluation checkpoint')
            if not (rows.bank_sha256==banks[f'{dataset}_test']['bank_sha256']).all():
                raise ValueError('Stale evaluation bank')
            if not (rows.manifest_sha256==identity['manifests'][f'{dataset}_test']).all():
                raise ValueError('Stale evaluation manifest')
            bank_metadata=pd.read_csv(bank_path(c,dataset,'test')/'metadata.csv')
            keys=['image_id','mask_id','condition','pattern','size','actual_ratio']
            observed=rows[keys].sort_values(['image_id','condition']).reset_index(drop=True)
            locked=bank_metadata[keys].sort_values(['image_id','condition']).reset_index(drop=True)
            pd.testing.assert_frame_equal(observed,locked,check_dtype=False,rtol=1e-9,atol=1e-10)
        from src.evaluation.evaluate import aggregate
        calculated=aggregate(per).sort_values(['model','eval_dataset','condition']).reset_index(drop=True)
        measured=summary.sort_values(['model','eval_dataset','condition']).reset_index(drop=True)
        pd.testing.assert_frame_equal(calculated,measured,check_dtype=False,rtol=1e-8,atol=1e-10)
    verify('verified_results',verify_results)
    verify('failure_cases',lambda:(out/'metrics/qualitative_selections.csv').is_file()
           and len(pd.read_csv(out/'metrics/qualitative_selections.csv'))==72
           and all((out/'figures'/f'{dataset}_{condition}_failure_grid.png').is_file()
                   for dataset in ['imagenette','pets'] for condition in CONDITIONS))
    verify('measured_report',lambda:project_path('report/results.md').is_file())
    if write_status:
        out.mkdir(parents=True,exist_ok=True)
        (out/'project_status.json').write_text(json.dumps({'checks':status,'errors':errors},indent=2),encoding='utf-8')
    print(json.dumps({'checks':status,'errors':errors},indent=2))
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--config',default='configs/base.yaml')
    a=p.parse_args()
    status=check(a.seed,a.config)
    raise SystemExit(0 if all(status.values()) else 1)
