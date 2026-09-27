"""Matched fixed masks: metadata existence alone never implies a valid bank."""
import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label
from src.utils.config import project_path, file_sha256
from src.utils.seed import stable_seed
from .generators import CONDITIONS, VERSION, generate_mask

def validate_bank(manifest, out, dataset, split, seed):
    out = project_path(out)
    with project_path(manifest).open(encoding='utf-8') as f:
        images = list(csv.DictReader(f))
    with (out / 'metadata.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    expected = {(r['image_id'], c) for r in images for c in CONDITIONS}
    if not expected or len(rows) != len(expected) or {(r['image_id'], r['condition']) for r in rows} != expected:
        raise ValueError('Bank has missing, duplicate or unexpected image/condition pairs')
    manifest_hash = file_sha256(manifest)
    groups, content = {}, hashlib.sha256()
    ids = set()
    for row in sorted(rows, key=lambda r: (r['image_id'], r['condition'])):
        condition, q = row['condition'], int(row['condition'][1:]) / 100
        expected_seed = stable_seed(seed, dataset, split, row['image_id'], condition[1:])
        if (row['dataset'] != dataset or row['split'] != split or row['pattern'] != condition[0]
                or row['generator_version'] != VERSION or row.get('manifest_sha256') != manifest_hash
                or int(row['mask_seed']) != expected_seed or float(row['target_ratio']) != q
                or int(row['size']) != int(condition[1:]) or row['mask_id'] in ids):
            raise ValueError('Stale/incompatible bank metadata; use a new mask_bank_dir')
        ids.add(row['mask_id'])
        if not 1 <= int(row['attempt_count']) <= 2000:
            raise ValueError('Invalid attempt count')
        path = (out / row['mask_path']).resolve()
        if not path.is_relative_to(out.resolve()) or path.suffix.lower() != '.png':
            raise ValueError('Invalid bank mask path')
        raw_hash = file_sha256(path)
        if raw_hash != row.get('mask_sha256'):
            raise ValueError(f'Mask content changed: {row["mask_id"]}')
        with Image.open(path) as image:
            mask = np.asarray(image)
        if mask.shape != (128,128) or not np.isin(mask,[0,255]).all():
            raise ValueError('Mask must be binary 128x128 PNG')
        regenerated, generated = generate_mask(condition[0],q,expected_seed)
        if not np.array_equal(regenerated,mask > 0):
            raise ValueError('Mask inconsistent with locked seed/generator')
        for key in ['actual_ratio','area_reference_ratio','area_match_tolerance',
                    'rectangle_width','rectangle_height','attempt_count']:
            if abs(float(row[key])-float(generated[key])) > 1e-9:
                raise ValueError(f'Invalid mask metadata: {key}')
        if condition[0] == 'H' and label(mask > 0)[1] != 6:
            raise ValueError('Multiple holes must have six components')
        groups.setdefault((row['image_id'],row['size']),{})[condition[0]] = mask > 0
        content.update(f'{row["image_id"]}|{condition}|{raw_hash}\n'.encode())
    for masks in groups.values():
        areas = [m.mean() for m in masks.values()]
        if max(areas)-min(areas) > .01:
            raise ValueError('Pattern areas differ by > one percentage point')
        def geometry(mask):
            y,x = np.where(mask)
            return int(x.max()-x.min()+1),int(y.max()-y.min()+1)
        if geometry(masks['C']) != geometry(masks['R']):
            raise ValueError('C/R rectangle geometry differs')
    content.update(file_sha256(out/'metadata.csv').encode())
    return {'manifest_sha256':manifest_hash,'bank_sha256':content.hexdigest(),
            'generator_version':VERSION,'mask_seed':int(seed),'n_pairs':len(rows)}

def build_bank(manifest, out, dataset, split, seed):
    out = project_path(out)
    if (out/'metadata.csv').exists():
        validated = validate_bank(manifest,out,dataset,split,seed)
        print(f'Validated locked bank {out}: {validated["bank_sha256"]}')
        with (out/'metadata.csv').open(encoding='utf-8') as f:
            return list(csv.DictReader(f))
    with project_path(manifest).open(encoding='utf-8') as f:
        images = list(csv.DictReader(f))
    if not images:
        raise ValueError('Cannot build a bank from an empty manifest')
    out.mkdir(parents=True,exist_ok=True)
    manifest_hash,metadata = file_sha256(manifest),[]
    for image in images:
        for condition in CONDITIONS:
            mask_seed = stable_seed(seed,dataset,split,image['image_id'],condition[1:])
            mask,meta = generate_mask(condition[0],int(condition[1:])/100,mask_seed)
            mask_id = f'{image["image_id"]}_{condition}_v{VERSION}'
            filename = mask_id+'.png'
            Image.fromarray(mask*255).save(out/filename)
            meta.update(dataset=dataset,split=split,image_id=image['image_id'],mask_id=mask_id,
                        condition=condition,size=int(condition[1:]),mask_path=filename,
                        mask_sha256=file_sha256(out/filename),manifest_sha256=manifest_hash)
            metadata.append(meta)
    temporary = out/'metadata.tmp'
    with temporary.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(metadata[0]))
        writer.writeheader(); writer.writerows(metadata)
    temporary.replace(out/'metadata.csv')
    validate_bank(manifest,out,dataset,split,seed)
    print(f'{dataset}/{split}: {len(metadata)} fixed, area-matched masks')
    return metadata

def validate_generator(seeds=100):
    for q in (.15,.30,.50):
        for seed in range(seeds):
            masks = {p:generate_mask(p,q,seed)[0] for p in 'CRFH'}
            areas = [m.mean() for m in masks.values()]
            if max(areas)-min(areas) > .01:
                raise ValueError('Unmatched pattern areas')
            for p,mask in masks.items():
                if mask.shape != (128,128) or not np.isin(mask,[0,1]).all() or abs(mask.mean()-q) > .02:
                    raise ValueError('Invalid generated mask')
                if p == 'H' and label(mask)[1] != 6:
                    raise ValueError('Invalid hole components')
    print(f'Validated {seeds} seeds x 12 area-matched conditions')
