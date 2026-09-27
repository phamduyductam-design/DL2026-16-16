"""Binary missing-region masks. C/R share geometry for the same seed and ratio."""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

VERSION = '2.0.0'
AREA_MATCH_TOLERANCE = .0049  # Max C/R/F/H area spread < one percentage point.
CONDITIONS = [f'{p}{q}' for p in 'CRFH' for q in (15, 30, 50)]

def rectangle_geometry(ratio, seed, size=128):
    rng = np.random.default_rng(seed)
    aspect = math.exp(rng.uniform(math.log(.75), math.log(1.33)))
    w = round(math.sqrt(ratio * size * size * aspect))
    h = round(math.sqrt(ratio * size * size / aspect))
    return w, h

def _free_form(rng, ratio, size):
    settings = {.15: (1, 3, 12, 36, 6, 16), .30: (2, 4, 16, 48, 10, 26),
                .50: (3, 5, 20, 56, 16, 38)}
    lo, hi, lmin, lmax, bmin, bmax = settings[ratio]
    canvas = Image.new('L', (size, size))
    draw = ImageDraw.Draw(canvas)
    for _ in range(int(rng.integers(lo, hi + 1))):
        x, y = rng.uniform(0, size - 1, 2)
        width = int(rng.integers(bmin, bmax + 1))
        radius = width / 2
        for _ in range(int(rng.integers(2, 6))):
            angle, length = rng.uniform(0, 2 * math.pi), rng.uniform(lmin, lmax)
            nx, ny = x + length * math.cos(angle), y + length * math.sin(angle)
            draw.line((x, y, nx, ny), fill=1, width=width)
            for px, py in ((x, y), (nx, ny)):
                draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=1)
            x, y = nx, ny
    return np.asarray(canvas, dtype=np.uint8).copy()

def generate_mask(pattern, ratio, seed, size=128, max_attempts=2000):
    if pattern not in 'CRFH' or pattern == '' or len(pattern) != 1 or ratio not in (.15, .30, .50):
        raise ValueError('Expected C/R/F/H and ratio .15/.30/.50')
    rng = np.random.default_rng(seed)
    for attempt in range(1, max_attempts + 1):
        mask = np.zeros((size, size), dtype=np.uint8)
        if pattern in 'CR':
            w, h = rectangle_geometry(ratio, seed, size)
            if pattern == 'C':
                x, y = (size-w)//2, (size-h)//2
            else:
                x, y = int(rng.integers(size-w+1)), int(rng.integers(size-h+1))
            mask[y:y+h, x:x+w] = 1
        elif pattern == 'F':
            mask = _free_form(rng, ratio, size)
        else:
            cols, rows = (3, 2) if rng.random() < .5 else (2, 3)
            for row in range(rows):
                for col in range(cols):
                    x0, x1 = col*size//cols, (col+1)*size//cols
                    y0, y1 = row*size//rows, (row+1)*size//rows
                    area = ratio*size*size/6
                    aspect = (x1-x0)/(y1-y0) * math.exp(rng.uniform(-.1, .1))
                    w = min(x1-x0-4, round(math.sqrt(area*aspect)))
                    h = min(y1-y0-4, round(area/w))
                    x = int(rng.integers(x0+1, x1-w))
                    y = int(rng.integers(y0+1, y1-h))
                    mask[y:y+h, x:x+w] = 1
            if label(mask)[1] != 6:
                continue
        actual = float(mask.mean())
        reference_w, reference_h = rectangle_geometry(ratio, seed, size)
        reference_ratio = reference_w * reference_h / (size * size)
        if abs(actual-ratio) <= .02 and abs(actual-reference_ratio) <= AREA_MATCH_TOLERANCE:
            return mask, {'pattern': pattern, 'target_ratio': ratio, 'actual_ratio': actual,
                          'mask_seed': int(seed), 'generator_version': VERSION, 'attempt_count': attempt,
                          'area_reference_ratio': reference_ratio, 'area_match_tolerance': AREA_MATCH_TOLERANCE,
                          'rectangle_width': reference_w, 'rectangle_height': reference_h}
    raise RuntimeError(f'Mask rejection cap reached: {pattern}{ratio}, seed={seed}, attempts={max_attempts}')
