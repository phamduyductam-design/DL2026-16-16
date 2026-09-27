from pathlib import Path
import numpy as np
from src.utils.config import project_path
from .imagenette import scan

def balanced_pet_selection(rows, seed):
    classes = sorted({r['class'] for r in rows},key=int)
    if len(classes) != 37:
        raise ValueError('Pet official test must cover all 37 breeds')
    rng = np.random.default_rng(seed+1)
    extras = set(rng.permutation(classes)[:19])
    selected = []
    for cls in classes:
        candidates = sorted([r for r in rows if r['class'] == cls],key=lambda r:r['path'])
        rng.shuffle(candidates)
        required = 14 if cls in extras else 13
        if len(candidates) < required:
            raise ValueError(f'Insufficient valid Pet test images for breed {cls}')
        selected.extend(candidates[:required])
    rng.shuffle(selected)
    return selected

def select_official_test(root, seed, near_index=None, near_duplicates=None):
    root = project_path(root)
    rows, invalid, duplicates = scan(root / 'images',near_index,near_duplicates)
    by_stem = {Path(r['path']).stem: r for r in rows}
    selected = []
    for line in (root / 'annotations' / 'test.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        stem, label, *_ = line.split()
        if stem in by_stem:
            row = dict(by_stem[stem])
            row['class'], row['path'] = label, 'images/' + row['path']
            selected.append(row)
    return balanced_pet_selection(selected,seed), invalid, duplicates
