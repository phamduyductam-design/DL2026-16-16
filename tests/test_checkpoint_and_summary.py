import pandas as pd
import torch
import random
import numpy as np
import pytest
from src.utils.seed import seed_everything,random_states,restore_states,stable_seed
from src.utils.checkpoint import save_checkpoint,load_checkpoint
from src.evaluation.evaluate import aggregate

def test_rng_checkpoint_resume(tmp_path):
    seed_everything(42)
    states=random_states()
    expected=(random.random(),np.random.rand(),torch.rand(3))
    save_checkpoint(tmp_path/'last.pt',{'random_states':states,'epoch':2})
    restore_states(load_checkpoint(tmp_path/'last.pt')['random_states'])
    assert random.random()==expected[0]
    assert np.random.rand()==expected[1]
    assert torch.equal(torch.rand(3),expected[2])

def test_per_image_condition_average():
    rows=[]
    for value in [.1,.3]:
        rows.append(dict(train_dataset='imagenette',eval_dataset='pets',model='unet',train_seed=42,
                         pattern='F',size=30,condition='F30',actual_ratio=.30,MAE_hole=value,
                         MSE_hole=value**2,PSNR_hole=10,PSNR_full=15,SSIM_full=.7,
                         inference_ms=2,parameter_count=10))
    summary=aggregate(pd.DataFrame(rows))
    assert len(summary)==1
    assert np.isclose(summary.MAE_hole.iloc[0],.2)
    assert np.isclose(summary.MSE_hole.iloc[0],.05)
    assert summary.n_pairs.iloc[0]==2

def test_model_independent_batch_order():
    orders=[]
    for model in ['autoencoder','unet']:
        seed_everything(42)
        from src.models import make_model
        make_model(model)
        generator=torch.Generator().manual_seed(stable_seed(42,'batch_order',0))
        orders.append(torch.randperm(6000,generator=generator))
    assert torch.equal(*orders)

def checkpoint_payload(epoch,value,best_epoch=None,best_metric=None):
    return {'model_state_dict':{'weight':torch.tensor([value])},'epoch':epoch,
            'best_epoch':epoch if best_epoch is None else best_epoch,
            'best_metric':value if best_metric is None else best_metric,'config':{'model':'test'},
            'data_identity':{'manifest':'fixed','bank':'fixed'},'run_info':{},'checkpoint_kind':'last'}

def test_resume_repairs_stale_and_missing_best(tmp_path):
    from src.utils.checkpoint import commit_training_checkpoints,repair_best_checkpoint
    last,best=tmp_path/'last.pt',tmp_path/'best.pt'
    first=commit_training_checkpoints(last,best,checkpoint_payload(1,.5))
    second=commit_training_checkpoints(last,best,checkpoint_payload(2,.2),first)
    save_checkpoint(best,first)  # Simulate stale BEST / a legacy interrupted order.
    repaired=repair_best_checkpoint(load_checkpoint(last),best)
    assert repaired['best_epoch']==2
    assert torch.equal(load_checkpoint(best)['model_state_dict']['weight'],torch.tensor([.2]))
    best.unlink()
    repair_best_checkpoint(load_checkpoint(last),best)
    assert best.exists()
    best.write_bytes(b'corrupted checkpoint')
    repair_best_checkpoint(load_checkpoint(last),best)
    assert load_checkpoint(best)['best_epoch']==2

def test_crash_between_best_and_last_rolls_back_best(tmp_path,monkeypatch):
    import src.utils.checkpoint as module
    last,best=tmp_path/'last.pt',tmp_path/'best.pt'
    first=module.commit_training_checkpoints(last,best,checkpoint_payload(1,.5))
    save=module.save_checkpoint
    def interrupted(path,payload):
        if path==last: raise OSError('simulated interruption')
        save(path,payload)
    monkeypatch.setattr(module,'save_checkpoint',interrupted)
    with pytest.raises(OSError): module.commit_training_checkpoints(last,best,checkpoint_payload(2,.2),first)
    monkeypatch.setattr(module,'save_checkpoint',save)
    assert load_checkpoint(best)['best_epoch']==2
    module.repair_best_checkpoint(load_checkpoint(last),best)
    assert load_checkpoint(best)['best_epoch']==1

def test_pipeline_respects_custom_output_dir(tmp_path,monkeypatch):
    import importlib.util
    import sys
    from src.utils.config import ROOT
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('pipeline_under_test',ROOT/'scripts/run_experiment.py')
    pipeline=importlib.util.module_from_spec(spec); spec.loader.exec_module(pipeline)
    output=tmp_path/'custom-output'
    checkpoints=output/'checkpoints'; checkpoints.mkdir(parents=True)
    (checkpoints/'autoencoder_seed42_last.pt').touch()
    manifests=tmp_path/'manifests'; manifests.mkdir()
    (manifests/'imagenette_train.csv').touch()
    def configs(path):
        if path=='data-config': return {'manifest_dir':str(manifests)}
        return {'output_dir':str(output),'dataset_config':'data-config','batch_size':16,'mask_bank_dir':str(tmp_path/'bank')}
    monkeypatch.setattr(pipeline,'load_config',configs)
    calls=[]; monkeypatch.setattr(pipeline,'run',lambda *args:calls.append(args))
    monkeypatch.setattr(sys,'argv',['run_experiment.py','--config','custom.yaml'])
    pipeline.main()
    training=[call for call in calls if call[0]=='scripts/train.py']
    assert training[0][-2:]==('--resume',str(checkpoints/'autoencoder_seed42_last.pt'))
    evaluations=[call for call in calls if call[0]=='scripts/evaluate.py' and '--checkpoint' in call]
    assert all(str(checkpoints) in call[call.index('--checkpoint')+1] for call in evaluations)
    assert all('--output' in call and str(output) in call for call in calls if '--combine' in call or call[0]=='scripts/plot_results.py')

def test_check_project_returns_failure_for_missing_artifacts(tmp_path,monkeypatch):
    import subprocess
    import sys
    import yaml
    from src.utils.config import ROOT
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'dataset_config':str(ROOT/'configs/datasets.yaml'),'output_dir':str(tmp_path/'out'),
                                    'mask_bank_dir':str(tmp_path/'missing-bank'),'mask_seed':314159,'train_seed':42,
                                    'epochs':30,'early_stopping_patience':5}))
    result=subprocess.run([sys.executable,str(ROOT/'scripts/check_project.py'),'--config',str(config)],
                           cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==1
    assert 'false' in result.stdout

def test_resume_rejects_changed_data_identity(tmp_path,monkeypatch):
    from src.training import engine
    from src.utils.config import load_config
    config=load_config('configs/base.yaml')
    config.update(device='cpu',output_dir=str(tmp_path/'outputs'))
    checkpoint=tmp_path/'last.pt'
    save_checkpoint(checkpoint,{'config':dict(config,model='autoencoder'),'data_identity':{'dataset':'old'}})
    monkeypatch.setattr(engine,'experiment_identity',lambda c:{'dataset':'changed'})
    with pytest.raises(ValueError,match='fingerprints differ'):
        engine.train(config,'autoencoder',checkpoint)
