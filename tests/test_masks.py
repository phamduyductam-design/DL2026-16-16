import numpy as np
import pytest
from scipy.ndimage import label
from src.masks.generators import CONDITIONS,generate_mask,rectangle_geometry

@pytest.mark.parametrize('condition',CONDITIONS)
def test_100_seeds(condition):
    for seed in range(100):
        mask,meta=generate_mask(condition[0],int(condition[1:])/100,seed)
        assert mask.shape==(128,128)
        assert set(np.unique(mask))=={0,1}
        assert abs(mask.mean()-int(condition[1:])/100)<=.02
        if condition[0]=='H': assert label(mask)[1]==6
        again,_=generate_mask(condition[0],int(condition[1:])/100,seed)
        assert np.array_equal(mask,again)
        assert meta['attempt_count']<=2000

@pytest.mark.parametrize('ratio',[.15,.30,.50])
def test_paired_rectangle(ratio):
    for seed in range(100):
        c,_=generate_mask('C',ratio,seed)
        r,_=generate_mask('R',ratio,seed)
        def bounds(m):
            y,x=np.where(m)
            return x.max()-x.min()+1,y.max()-y.min()+1
        assert bounds(c)==bounds(r)==rectangle_geometry(ratio,seed)
        assert c.sum()==r.sum()

def test_attempt_cap():
    with pytest.raises(RuntimeError): generate_mask('F',.5,0,max_attempts=0)

@pytest.mark.parametrize('ratio',[.15,.30,.50])
def test_matched_pattern_areas(ratio):
    for seed in range(100):
        areas=[generate_mask(pattern,ratio,seed)[0].mean() for pattern in 'CRFH']
        assert max(areas)-min(areas)<=.01

def test_bank_validates_existing_files_and_manifest(tmp_path):
    import csv
    from src.masks.bank import build_bank,validate_bank
    manifest=tmp_path/'manifest.csv'
    manifest.write_text('image_id,path,class,sha256\nimage,sample.jpg,cat,digest\n')
    bank=tmp_path/'bank'
    build_bank(manifest,bank,'pets','test',314159)
    identity=validate_bank(manifest,bank,'pets','test',314159)
    assert identity['n_pairs']==12
    assert len(build_bank(manifest,bank,'pets','test',314159))==12
    with (bank/'metadata.csv').open() as f: rows=list(csv.DictReader(f))
    mask=bank/rows[0]['mask_path']
    original=mask.read_bytes()
    mask.write_bytes(b'changed')
    with pytest.raises(ValueError,match='content changed'):
        validate_bank(manifest,bank,'pets','test',314159)
    mask.write_bytes(original)
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='Stale'):
        build_bank(manifest,bank,'pets','test',314159)

def test_bank_rejects_old_generator_version(tmp_path):
    import csv
    from src.masks.bank import build_bank
    manifest=tmp_path/'manifest.csv'
    manifest.write_text('image_id,path,class,sha256\nimage,sample.jpg,cat,digest\n')
    bank=tmp_path/'bank'
    build_bank(manifest,bank,'pets','test',314159)
    with (bank/'metadata.csv').open() as f: rows=list(csv.DictReader(f))
    rows[0]['generator_version']='1.0.0'
    with (bank/'metadata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError,match='Stale'):
        build_bank(manifest,bank,'pets','test',314159)
