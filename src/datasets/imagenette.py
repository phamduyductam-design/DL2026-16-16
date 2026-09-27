"""Folder and tgz readers; archive contents are cached without extracting files."""
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dctn
import torch
from torch.utils.data import Dataset
from src.utils.config import project_path, file_sha256
from src.utils.seed import stable_seed

@lru_cache(maxsize=2)
def archive_images(root):
    result = {}
    with tarfile.open(root, 'r:*') as archive:
        for member in archive:
            if member.isfile() and Path(member.name).suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                result[member.name] = archive.extractfile(member).read()
    return result

def read_bytes(root, relative):
    root = project_path(root)
    return archive_images(str(root))[relative] if root.is_file() else (root / relative).read_bytes()

DATA_PROTOCOL = 'v2-train-val-from-official-train-balanced-pets'

class NearDuplicateIndex:
    """Conservative pHash candidate search plus RGB thumbnail error confirmation."""
    def __init__(self):
        self.entries, self.buckets = [], {}
    def match_or_add(self, image, image_id):
        thumb = np.asarray(image.convert('RGB').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)/255
        gray = np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
        frequency = dctn(gray,type=2,norm='ortho')[:8,:8].ravel()
        bits = frequency > np.median(frequency[1:])
        phash = sum(int(b)<<i for i,b in enumerate(bits))
        keys = [(i,(phash >> start)&((1<<width)-1)) for i,(start,width) in
                enumerate([(0,13),(13,13),(26,13),(39,13),(52,12)])]
        candidates = set()
        for key in keys:
            candidates.update(self.buckets.get(key,[]))
        for index in sorted(candidates):
            other_hash,other_thumb,other_id = self.entries[index]
            distance = (phash^other_hash).bit_count()
            if distance <= 4:
                error = thumb-other_thumb
                mae,mse = float(np.abs(error).mean()),float(np.square(error).mean())
                if mae <= .04 and mse <= .003:
                    return {'image_id':image_id,'duplicate_of':other_id,'phash_distance':distance,
                            'thumbnail_MAE':mae,'thumbnail_MSE':mse}
        index = len(self.entries)
        self.entries.append((phash,thumb,image_id))
        for key in keys:
            self.buckets.setdefault(key,[]).append(index)
        return None

def scan(root, near_index=None, near_duplicates=None):
    root = project_path(root)
    paths = sorted(archive_images(str(root))) if root.is_file() else sorted(
        p.relative_to(root).as_posix() for p in root.rglob('*')
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    valid, invalid, duplicates, seen = [], [], [], set()
    for path in paths:
        raw = read_bytes(root, path)
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                image.convert('RGB').load()
        except (OSError, ValueError) as error:
            invalid.append({'path': path, 'error': str(error)})
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            duplicates.append(path)
            continue
        if near_index is not None:
            with Image.open(io.BytesIO(raw)) as image:
                match = near_index.match_or_add(image,digest)
            if match is not None:
                if near_duplicates is not None:
                    near_duplicates.append(dict(match,path=path))
                continue
        seen.add(digest)
        valid.append({'image_id': digest, 'path': path, 'class': Path(path).parent.name, 'sha256': digest})
    return valid, invalid, duplicates

def stratified_imagenette(rows, seed):
    rng = np.random.default_rng(seed)
    classes = sorted({r['class'] for r in rows})
    if len(classes) != 10:
        raise ValueError(f'Expected 10 Imagenette classes, got {classes}')
    splits = {'imagenette_train': [], 'imagenette_val': [], 'imagenette_test': []}
    for cls in classes:
        train = [r for r in rows if r['class'] == cls and 'train' in Path(r['path']).parts]
        heldout = [r for r in rows if r['class'] == cls and 'val' in Path(r['path']).parts]
        if len(train) < 630 or len(heldout) < 100:
            raise ValueError(f'Insufficient valid images for {cls}: {len(train)}, {len(heldout)}')
        rng.shuffle(train)
        rng.shuffle(heldout)
        splits['imagenette_train'].extend(train[:600])
        splits['imagenette_val'].extend(train[600:630])
        splits['imagenette_test'].extend(heldout[:100])
    return splits

def prepare_manifests(config):
    from .pets import select_official_test
    out = project_path(config['manifest_dir'])
    names = ['imagenette_train', 'imagenette_val', 'imagenette_test', 'pets_test']
    if any((out / f'{n}.csv').exists() for n in names):
        raise FileExistsError('Manifest already exists. Splits are locked; use another manifest_dir for a new experiment.')
    near_index,near_duplicates = NearDuplicateIndex(),[]
    rows, invalid, duplicates = scan(config['imagenette']['root'],near_index,near_duplicates)
    splits = stratified_imagenette(rows, config['split_seed'])
    pets, pet_invalid, pet_duplicates = select_official_test(config['pets']['root'], config['split_seed'],
                                                           near_index,near_duplicates)
    splits['pets_test'] = pets
    ids = [r['sha256'] for split in splits.values() for r in split]
    if len(ids) != len(set(ids)):
        raise ValueError('Content overlap detected across splits/datasets')
    out.mkdir(parents=True, exist_ok=True)
    summary = {'counts': {}, 'class_counts': {}, 'invalid': invalid + pet_invalid,
               'duplicates': duplicates + pet_duplicates, 'config': config,
               'protocol': DATA_PROTOCOL, 'near_duplicates': near_duplicates,
               'near_duplicate_policy': {'phash_hamming_max':4,'thumbnail_size':32,
                   'RGB_MAE_max':.04,'RGB_MSE_max':.003,'scope':'all scanned sources; deterministic deduplication',
                   'limitation':'Approximate screening; not proof that all semantic near-duplicates are absent'},
               'manifest_sha256':{}}
    for name, selected in splits.items():
        with (out / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_id', 'path', 'class', 'sha256'])
            writer.writeheader()
            writer.writerows(selected)
        summary['counts'][name] = len(selected)
        summary['class_counts'][name] = dict(Counter(r['class'] for r in selected))
        summary['manifest_sha256'][name] = file_sha256(out/f'{name}.csv')
    (out / 'audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

def validate_manifests(config, verify_images=True):
    """Validate protocol, locked CSVs, source membership, hashes and near duplicates."""
    out = project_path(config['manifest_dir'])
    audit = json.loads((out/'audit.json').read_text(encoding='utf-8'))
    if audit.get('protocol') != DATA_PROTOCOL or audit.get('config',{}).get('split_seed') != config['split_seed']:
        raise ValueError('Old/incompatible data protocol; prepare fresh manifests before training')
    expected = {'imagenette_train':6000,'imagenette_val':300,'imagenette_test':1000,'pets_test':500}
    all_ids,frames = set(),{}
    near_index = NearDuplicateIndex()
    pet_root = project_path(config['pets']['root'])
    pet_labels = {}
    if verify_images:
        for line in (pet_root/'annotations/test.txt').read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                stem,cls,*_ = line.split()
                pet_labels[stem] = cls
    for name,count in expected.items():
        path = out/f'{name}.csv'
        if file_sha256(path) != audit.get('manifest_sha256',{}).get(name):
            raise ValueError(f'Locked manifest changed: {name}')
        with path.open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        frames[name] = rows
        if len(rows) != count:
            raise ValueError(f'Wrong split count: {name}')
        classes = Counter(r['class'] for r in rows)
        if audit.get('counts',{}).get(name) != len(rows) or audit.get('class_counts',{}).get(name) != dict(classes):
            raise ValueError(f'Audit counts do not match manifest: {name}')
        if name.startswith('imagenette'):
            per_class = count//10
            if len(classes) != 10 or set(classes.values()) != {per_class}:
                raise ValueError(f'Unbalanced split: {name}')
            official = 'val' if name == 'imagenette_test' else 'train'
            if any(official not in Path(r['path']).parts for r in rows):
                raise ValueError(f'Incorrect official source: {name}')
            if any(Path(r['path']).parent.name != r['class'] for r in rows):
                raise ValueError(f'Imagenette class does not match source path: {name}')
        elif len(classes) != 37 or sorted(classes.values()) != [13]*18+[14]*19:
            raise ValueError('Pet requires 13-14 images per breed')
        for row in rows:
            digest = row['sha256']
            if row['image_id'] != digest or digest in all_ids or len(digest) != 64:
                raise ValueError('Overlapping/invalid image IDs')
            all_ids.add(digest)
            if verify_images:
                dataset = 'pets' if name == 'pets_test' else 'imagenette'
                if dataset == 'pets' and pet_labels.get(Path(row['path']).stem) != row['class']:
                    raise ValueError('Pet image outside official test split or wrong label')
                raw = read_bytes(config[dataset]['root'],row['path'])
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError(f'Changed source image: {row["path"]}')
                with Image.open(io.BytesIO(raw)) as image:
                    image.convert('RGB').load()
                    if near_index.match_or_add(image,digest) is not None:
                        raise ValueError('Near duplicate found among selected manifest images')
    if {r['class'] for r in frames['imagenette_train']} != {r['class'] for r in frames['imagenette_test']}:
        raise ValueError('Imagenette classes differ across splits')
    return {n:len(r) for n,r in frames.items()}

def preprocess(image, train=False, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    image = image.convert('RGB')
    w, h = image.size
    scale = 144 / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)
    w, h = image.size
    x = int(rng.integers(w - 128 + 1)) if train else (w - 128) // 2
    y = int(rng.integers(h - 128 + 1)) if train else (h - 128) // 2
    image = image.crop((x, y, x + 128, y + 128))
    if train and rng.random() < .5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1) / 255)

class InpaintingDataset(Dataset):
    def __init__(self, root, manifest, train=False, bank=None, train_seed=42, mask_seed=314159):
        self.root = root
        with project_path(manifest).open(encoding='utf-8') as f:
            self.rows = list(csv.DictReader(f))
        self.train, self.bank = train, bank
        self.train_seed, self.mask_seed, self.epoch = train_seed, mask_seed, 0
        self.fixed = None
        if bank is not None:
            with (project_path(bank) / 'metadata.csv').open(encoding='utf-8') as f:
                self.fixed = list(csv.DictReader(f))
            self.by_id = {r['image_id']: r for r in self.rows}
            if len(self.fixed) != 12 * len(self.rows):
                raise ValueError('Mask bank does not match manifest size')
            if {r['image_id'] for r in self.fixed} != set(self.by_id):
                raise ValueError('Mask bank image IDs differ from manifest')
            from src.masks.generators import CONDITIONS
            pairs = {(r['image_id'],r['condition']) for r in self.fixed}
            if len(pairs) != len(self.fixed) or pairs != {(i,c) for i in self.by_id for c in CONDITIONS}:
                raise ValueError('Mask bank must contain each of the 12 conditions exactly once per image')
    def __len__(self):
        return len(self.fixed) if self.fixed is not None else len(self.rows)
    def __getitem__(self, index):
        from src.masks.generators import CONDITIONS, generate_mask
        meta = dict(self.fixed[index]) if self.fixed is not None else {}
        row = self.by_id[meta['image_id']] if self.fixed is not None else self.rows[index]
        rng = np.random.default_rng(stable_seed(self.train_seed, self.epoch, row['image_id'], 'augment'))
        raw = read_bytes(self.root, row['path'])
        if 'sha256' in row and row['sha256'] and row['sha256'] != hashlib.sha256(raw).hexdigest():
            raise ValueError(f'Image contents changed after manifest lock: {row["path"]}')
        with Image.open(io.BytesIO(raw)) as image:
            tensor = preprocess(image, self.train, rng)
        if self.fixed is not None:
            with Image.open(project_path(self.bank) / meta['mask_path']) as mask_image:
                mask = np.asarray(mask_image, dtype=np.float32) / 255
            if mask.shape != (128,128) or not np.isin(mask,[0,1]).all():
                raise ValueError(f'Invalid binary mask: {meta["mask_id"]}')
            if abs(float(mask.mean())-float(meta['actual_ratio'])) > 1e-7:
                raise ValueError(f'Mask changed after bank lock: {meta["mask_id"]}')
        else:
            condition = CONDITIONS[(index + self.epoch) % 12]
            seed = stable_seed(self.mask_seed, self.epoch, row['image_id'], condition[1:])
            mask, meta = generate_mask(condition[0], int(condition[1:]) / 100, seed)
            meta.update(mask_id=f'train-{self.epoch}-{row["image_id"]}-{condition}', condition=condition)
        return {'image': tensor, 'mask': torch.from_numpy(mask.copy()).float().unsqueeze(0),
                'image_id': row['image_id'], 'path': row['path'], 'class': row['class'],
                'mask_id': meta['mask_id'], 'condition': meta['condition'],
                'actual_ratio': float(meta['actual_ratio'])}
