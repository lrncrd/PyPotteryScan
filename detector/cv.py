"""K-fold cross-validation of the detector, plus the final fit on all pages.

Pools the train+val images made by prepare_dataset.py and splits them in K folds, each
project spread evenly over the folds (page i of a project -> fold i % K), so every fold
covers every context. Each fold trains on K-1 folds and is scored on the held-out one,
overall and per project. Fold datasets are symlinks: nothing is copied.

Usage:
    python detector/prepare_dataset.py
    python detector/cv.py --folds 5                  # all folds, then summary
    python detector/cv.py --folds 5 --fold 2         # a single fold
    python detector/cv.py --final --epochs 100       # train on ALL pages -> runs/final_*
    python detector/cv.py --summary                  # re-print the table from saved JSON
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATASET = HERE / 'dataset'
CV_DIR = HERE / 'cv'
RUNS = HERE / 'runs'
NAMES = {0: 'drawing', 1: 'text'}


def pooled_pages():
    """[(project_label, image_path, label_path)], sorted so fold assignment is stable."""
    sys.path.insert(0, str(HERE))
    from prepare_dataset import PROJECTS_DIR, project_label
    labels = sorted({project_label(d) for d in PROJECTS_DIR.iterdir() if d.is_dir()},
                    key=len, reverse=True)  # longest first: Birilai_1 before any shorter prefix
    pages = []
    for split in ('train', 'val'):
        for img in sorted((DATASET / 'images' / split).glob('*.jpg')):
            label = next(l for l in labels if img.stem.startswith(l + '_'))
            label = re.sub(r'_\d{8}$', '', label)  # Montale_20261003 -> Montale (display only)
            pages.append((label, img, DATASET / 'labels' / split / f'{img.stem}.txt'))
    pages.sort(key=lambda p: (p[0], p[1].name))
    return pages


def link_split(root, split, pages):
    for _, img, lab in pages:
        for kind, src, ext in (('images', img, '.jpg'), ('labels', lab, '.txt')):
            d = root / kind / split
            d.mkdir(parents=True, exist_ok=True)
            link = d / (img.stem + ext)
            if not link.exists():
                link.symlink_to(src)


def write_yaml(root, train_list_ok=True):
    y = {'path': str(root), 'train': 'images/train', 'val': 'images/val', 'names': NAMES}
    (root / 'data.yaml').write_text(yaml.safe_dump(y))
    return root / 'data.yaml'


def metrics_of(model, data, imgsz, device, name):
    r = model.val(data=str(data), imgsz=imgsz, batch=4, device=device, plots=False,
                  verbose=False, project=str(CV_DIR / 'val'), name=name, exist_ok=True)
    b = r.box
    return {NAMES[i]: {'P': float(b.p[i]), 'R': float(b.r[i]),
                       'mAP50': float(b.ap50[i]), 'mAP50-95': float(b.ap[i])}
            for i in range(len(b.p))} | {'all': {'P': float(b.mp), 'R': float(b.mr),
                                                 'mAP50': float(b.map50), 'mAP50-95': float(b.map)}}


def train_kwargs(args):
    return dict(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                fliplr=0.0, flipud=0.0, degrees=3.0, mosaic=0.5, patience=args.patience,
                seed=0, deterministic=True, exist_ok=True)


def run_fold(args, k):
    pages = pooled_pages()
    idx = defaultdict(int)
    folds = []
    for p in pages:
        folds.append(idx[p[0]] % args.folds)
        idx[p[0]] += 1
    root = CV_DIR / f'fold{k}'
    for sub in ('images', 'labels'):
        for s in ('train', 'val'):
            d = root / sub / s
            if d.exists():
                for f in d.iterdir():
                    f.unlink()
    link_split(root, 'train', [p for p, f in zip(pages, folds) if f != k])
    link_split(root, 'val', [p for p, f in zip(pages, folds) if f == k])
    data = write_yaml(root)

    model = YOLO(args.weights)
    model.train(data=str(data), project=str(RUNS), name=f'cv_fold{k}', **train_kwargs(args))
    best = YOLO(str(RUNS / f'cv_fold{k}' / 'weights' / 'best.pt'))

    out = {'fold': k, 'overall': metrics_of(best, data, args.imgsz, args.device, f'fold{k}'),
           'per_project': {}}
    val_pages = [p for p, f in zip(pages, folds) if f == k]
    for proj in sorted({p[0] for p in val_pages}):
        lst = CV_DIR / f'fold{k}_{proj}.txt'
        lst.write_text('\n'.join(str(root / 'images' / 'val' / p[1].name)
                                 for p in val_pages if p[0] == proj) + '\n')
        y = {'path': str(root), 'train': str(lst), 'val': str(lst), 'names': NAMES}
        (CV_DIR / f'fold{k}_{proj}.yaml').write_text(yaml.safe_dump(y))
        out['per_project'][proj] = {
            'pages': sum(1 for p in val_pages if p[0] == proj),
            **metrics_of(best, CV_DIR / f'fold{k}_{proj}.yaml', args.imgsz, args.device,
                         f'fold{k}_{proj}')}
    (CV_DIR / f'fold{k}.json').write_text(json.dumps(out, indent=1))


def summary():
    res = [json.loads(f.read_text()) for f in sorted(CV_DIR.glob('fold*.json'))]
    if not res:
        print('no fold results')
        return
    print(f'{len(res)} fold(s)\n')
    print(f'{"":14s}{"mAP50":>16s}{"mAP50-95":>16s}{"P":>8s}{"R":>8s}')
    for cls in ('drawing', 'text', 'all'):
        row = {m: [r['overall'][cls][m] for r in res] for m in ('mAP50', 'mAP50-95', 'P', 'R')}
        f = lambda v: f'{st.mean(v):.3f}±{st.pstdev(v):.3f}'
        print(f'{cls:14s}{f(row["mAP50"]):>16s}{f(row["mAP50-95"]):>16s}'
              f'{st.mean(row["P"]):8.3f}{st.mean(row["R"]):8.3f}')
    print('\nper project (mean over folds; pages = total held-out pages)')
    projs = sorted({p for r in res for p in r['per_project']})
    for p in projs:
        rs = [r['per_project'][p] for r in res if p in r['per_project']]
        n = sum(x['pages'] for x in rs)
        d = st.mean(x['drawing']['mAP50'] for x in rs)
        t = st.mean(x['text']['mAP50'] for x in rs)
        flag = '  <-- text < 0.80' if t < 0.80 else ''
        print(f'  {p:14s} pages={n:3d}  drawing mAP50 {d:.3f}  text mAP50 {t:.3f}{flag}')


def run_final(args):
    pages = pooled_pages()
    root = CV_DIR / 'all'
    link_split(root, 'train', pages)
    link_split(root, 'val', pages[::7])  # only used for the per-epoch monitor, not for selection
    data = write_yaml(root)
    kw = train_kwargs(args)
    kw['patience'] = args.epochs  # no early stop: val is part of train here, so it can't select
    YOLO(args.weights).train(data=str(data), project=str(RUNS), name='final', **kw)
    print(f'final weights: {RUNS / "final" / "weights" / "last.pt"} (use last.pt, not best.pt)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolo11s.pt')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    CV_DIR.mkdir(exist_ok=True)
    if args.summary:
        return summary()
    if args.final:
        return run_final(args)
    for k in ([args.fold] if args.fold is not None else range(args.folds)):
        run_fold(args, k)
    summary()


if __name__ == '__main__':
    main()
