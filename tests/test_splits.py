import csv
import io
import numpy as np
from PIL import Image
from src.datasets.imagenette import stratified_imagenette,InpaintingDataset
from src.utils.config import ROOT

def test_deterministic_counts_no_overlap():
    rows=[]
    for c in range(10):
        for split,n in [('train',650),('val',140)]:
            rows += [{'class':str(c),'path':f'{split}/{c}/{i}.jpg','image_id':f'{split}-{c}-{i}'} for i in range(n)]
    result=stratified_imagenette(rows,2026)
    assert result==stratified_imagenette(rows,2026)
    assert [len(result[n]) for n in ['imagenette_train','imagenette_val','imagenette_test']]==[6000,300,1000]
    ids=[r['image_id'] for rows in result.values() for r in rows]
    assert len(ids)==len(set(ids))
    assert all(r['path'].startswith('train/') for r in result['imagenette_val'])
    assert all(r['path'].startswith('val/') for r in result['imagenette_test'])

def test_item_shape(tmp_path):
    Image.fromarray(np.full((160,220,3),128,dtype=np.uint8)).save(tmp_path/'sample.jpg')
    manifest=tmp_path/'manifest.csv'
    manifest.write_text('image_id,path,class,sha256\na,sample.jpg,cat,\n')
    dataset=InpaintingDataset(str(tmp_path),manifest,train=True)
    a,b=dataset[0],dataset[0]
    assert a['image'].shape==(3,128,128) and a['mask'].shape==(1,128,128)
    assert 0<=a['image'].min()<=a['image'].max()<=1
    assert np.array_equal(a['image'],b['image'])

def test_real_manifests_if_prepared():
    path=ROOT/'data/manifests'
    if not (path/'imagenette_train.csv').exists(): return
    ids=[]
    for name,count in [('imagenette_train',6000),('imagenette_val',300),('imagenette_test',1000),('pets_test',500)]:
        with (path/f'{name}.csv').open() as f: rows=list(csv.DictReader(f))
        assert len(rows)==count
        ids.extend(r['sha256'] for r in rows)
    assert len(ids)==len(set(ids))

def test_balanced_pet_selection():
    from collections import Counter
    from src.datasets.pets import balanced_pet_selection
    rows=[{'class':str(cls),'path':f'breed{cls}_{i}.jpg'} for cls in range(1,38) for i in range(30)]
    first=balanced_pet_selection(rows,2026)
    assert first==balanced_pet_selection(rows,2026)
    assert len(first)==500
    assert sorted(Counter(r['class'] for r in first).values())==[13]*18+[14]*19

def test_near_duplicates_are_detected_without_matching_bytes():
    from src.datasets.imagenette import NearDuplicateIndex
    rng=np.random.default_rng(19)
    image=Image.fromarray(rng.integers(0,256,(80,80,3),dtype=np.uint8))
    encoded=io.BytesIO(); image.save(encoded,format='JPEG',quality=95)
    decoded=Image.open(io.BytesIO(encoded.getvalue()))
    index=NearDuplicateIndex()
    assert index.match_or_add(image,'original') is None
    match=index.match_or_add(decoded,'jpeg-copy')
    assert match is not None and match['duplicate_of']=='original'
    assert index.match_or_add(Image.new('RGB',(80,80),'white'),'different') is None

def test_locked_manifest_detects_tampering(tmp_path):
    import shutil
    import pytest
    from src.datasets.imagenette import validate_manifests
    from src.utils.config import load_config
    for file in (ROOT/'data/manifests').iterdir():
        shutil.copy2(file,tmp_path/file.name)
    config=load_config('configs/datasets.yaml')
    config['manifest_dir']=str(tmp_path)
    assert validate_manifests(config,verify_images=False)['pets_test']==500
    with (tmp_path/'imagenette_val.csv').open('a') as f: f.write('\n')
    with pytest.raises(ValueError,match='manifest changed'):
        validate_manifests(config,verify_images=False)
