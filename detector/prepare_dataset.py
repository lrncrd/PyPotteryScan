"""Convert PyPotteryScan project annotations into an Ultralytics YOLO detection dataset.

Classes: 0 = drawing, 1 = text.

Annotations are stored in the coordinate frame the browser shows, which applies the
EXIF orientation tag; PIL does not, so every image goes through exif_transpose here.

Usage:
    python detector/prepare_dataset.py                       # mixed split: 1 page in 7 of every project -> val
    python detector/prepare_dataset.py --holdout Stromboli_1 # validate on one project only
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / 'projects'
OUT_DIR = ROOT / 'detector' / 'dataset'
MAX_SIDE = 1920  # downscale the huge scans once; YOLO boxes are normalized so nothing else changes
CLASSES = ['drawing', 'text']
VAL_EVERY = 7  # mixed split: one page in VAL_EVERY goes to validation


def project_label(project_dir):
    """'Birilai_1_20261003_140606' -> 'Birilai_1'"""
    return '_'.join(project_dir.name.split('_')[:2])


def yolo_line(cls_id, box, w, h):
    x0, y0 = max(0, box['x']), max(0, box['y'])
    x1, y1 = min(w, box['x'] + box['w']), min(h, box['y'] + box['h'])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return f"{cls_id} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', help="project label (e.g. Stromboli_1) used as the validation set")
    args = ap.parse_args()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    counts = {'train': [0, 0, 0], 'val': [0, 0, 0]}  # images, drawings, text
    warnings = []

    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        ann_dir = project_dir / 'annotations'
        img_dir = project_dir / 'original_images'
        if not ann_dir.is_dir() or not img_dir.is_dir():
            continue
        label = project_label(project_dir)
        split = 'val' if args.holdout == label else 'train'
        if args.holdout is None:
            split = None  # decided per-image below

        page_idx = 0
        images = {p.stem: p for p in img_dir.iterdir() if p.is_file()}
        for ann_file in sorted(ann_dir.glob('*_annotations.json')):
            stem = ann_file.name[:-len('_annotations.json')]
            if stem not in images:
                warnings.append(f"{label}/{stem}: no matching image")
                continue
            drawings = json.loads(ann_file.read_text(encoding='utf8')).get('drawings', [])
            if not drawings:
                continue

            img = ImageOps.exif_transpose(Image.open(images[stem])).convert('RGB')
            w, h = img.size
            lines = []
            for d in drawings:
                for cls_id, box in [(0, d)] + [(1, t) for t in d.get('textBoxes', [])]:
                    if box['x'] + box['w'] > w * 1.02 or box['y'] + box['h'] > h * 1.02:
                        warnings.append(f"{label}/{stem}: {CLASSES[cls_id]} box exceeds image {w}x{h}")
                    line = yolo_line(cls_id, box, w, h)
                    if line:
                        lines.append(line)

            this_split = split
            if this_split is None:
                # Mixed validation: every 7th annotated page of *each* project, so val covers
                # every context in proportion to training.
                this_split = 'val' if page_idx % VAL_EVERY == VAL_EVERY - 1 else 'train'
            page_idx += 1

            scale = MAX_SIDE / max(w, h)
            if scale < 1:
                img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            name = f"{label}_{stem}"
            (OUT_DIR / 'images' / this_split).mkdir(parents=True, exist_ok=True)
            (OUT_DIR / 'labels' / this_split).mkdir(parents=True, exist_ok=True)
            img.save(OUT_DIR / 'images' / this_split / f"{name}.jpg", quality=92)
            (OUT_DIR / 'labels' / this_split / f"{name}.txt").write_text('\n'.join(lines) + '\n')

            c = counts[this_split]
            c[0] += 1
            c[1] += sum(1 for l in lines if l.startswith('0 '))
            c[2] += sum(1 for l in lines if l.startswith('1 '))

    (OUT_DIR / 'data.yaml').write_text(
        f"path: {OUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + ''.join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    for split, (n_img, n_dr, n_tx) in counts.items():
        print(f"{split}: {n_img} images, {n_dr} drawings, {n_tx} text boxes")
    for w in warnings:
        print('WARNING', w)


if __name__ == '__main__':
    main()
