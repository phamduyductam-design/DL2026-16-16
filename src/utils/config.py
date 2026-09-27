from pathlib import Path
import hashlib
import yaml

ROOT = Path(__file__).resolve().parents[2]

def project_path(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p

def load_config(path):
    with project_path(path).open(encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if 'extends' in config:
        base = load_config(config.pop('extends'))
        base.update(config)
        config = base
    return config

def file_sha256(path):
    digest = hashlib.sha256()
    with project_path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def bank_path(config, dataset, split):
    return project_path(config.get('mask_bank_dir', 'data/mask_bank')) / dataset / split

def experiment_identity(config, splits=None, verify_images=True):
    """Hash manifests and validated mask contents; reject stale locked data."""
    from src.datasets.imagenette import validate_manifests
    from src.masks.bank import validate_bank
    datasets = load_config(config['dataset_config'])
    validate_manifests(datasets, verify_images=verify_images)
    manifest_dir = project_path(datasets['manifest_dir'])
    names = ['imagenette_train','imagenette_val','imagenette_test','pets_test']
    manifests = {n: file_sha256(manifest_dir / f'{n}.csv') for n in names}
    banks = {}
    splits = [('imagenette','val'),('imagenette','test'),('pets','test')] if splits is None else splits
    for dataset, split in splits:
        banks[f'{dataset}_{split}'] = validate_bank(manifest_dir / f'{dataset}_{split}.csv',
            bank_path(config,dataset,split), dataset, split, config['mask_seed'])
    return {'manifests': manifests, 'banks': banks, 'audit_sha256': file_sha256(manifest_dir / 'audit.json')}
