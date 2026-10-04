"""Compare NMS / box-merging settings on the validation split (precision / recall / F1 at IoU 0.5).

Fixes the "one label split into several boxes" problem: YOLO's NMS only looks at IoU, so a small
box inside a bigger one survives. Run after prepare_dataset.py + train.py:

    python detector/tune_postprocess.py [path/to/best.pt]
"""
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
VAL_IMAGES = HERE / 'dataset' / 'images' / 'val'
VAL_LABELS = HERE / 'dataset' / 'labels' / 'val'
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else str(HERE / 'runs' / 'scan_detector5' / 'weights' / 'best.pt')


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a, b):
    i = inter(a, b)
    return i / (area(a) + area(b) - i + 1e-9)


def union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def merge_contained(boxes, thr):
    """Merge same-class boxes whose overlap covers >= thr of the smaller one (repeat to a fixpoint)."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if inter(boxes[i], boxes[j]) / (min(area(boxes[i]), area(boxes[j])) + 1e-9) >= thr:
                    boxes[i] = union(boxes[i], boxes[j])
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def merge_stacked(boxes, gap_ratio, overlap=0.5):
    """Merge boxes that are vertically adjacent (gap < gap_ratio * the smaller height) and overlap
    horizontally by >= `overlap` of the narrower one: lines of one label printed as separate boxes."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                hx = min(a[2], b[2]) - max(a[0], b[0])
                narrow = min(a[2] - a[0], b[2] - b[0])
                vgap = max(a[1] - b[3], b[1] - a[3], 0)
                small_h = min(a[3] - a[1], b[3] - b[1])
                if hx >= overlap * narrow and vgap <= gap_ratio * small_h:
                    boxes[i] = union(a, b)
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def load_gt(label_file, w, h):
    gt = {0: [], 1: []}
    for line in label_file.read_text().split('\n'):
        if line.strip():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            gt[int(c)].append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return gt


def score(preds, gts, thr=0.5):
    """Greedy match by descending IoU. Returns (tp, fp, fn) summed over pages."""
    tp = fp = fn = 0
    for pred, gt in zip(preds, gts):
        used = set()
        for p in sorted(pred, key=lambda b: -area(b)):
            best, best_iou = None, thr
            for k, g in enumerate(gt):
                if k not in used and iou(p, g) >= best_iou:
                    best, best_iou = k, iou(p, g)
            if best is None:
                fp += 1
            else:
                used.add(best)
                tp += 1
        fn += len(gt) - len(used)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9)


def main():
    model = YOLO(WEIGHTS)
    files = sorted(VAL_IMAGES.glob('*.jpg'))
    gts = []
    for f in files:
        w, h = Image.open(f).size
        gts.append(load_gt(VAL_LABELS / f"{f.stem}.txt", w, h))

    configs = []
    for conf in (0.25, 0.4, 0.5):
        for nms_iou in (0.7, 0.5, 0.3):
            configs.append((conf, nms_iou))

    post = {
        'none': lambda b, c: b,
        'contain 0.6': lambda b, c: merge_contained(b, 0.6),
        'contain 0.8': lambda b, c: merge_contained(b, 0.8),
        'contain 0.6 + stack text 0.3': lambda b, c: merge_stacked(merge_contained(b, 0.6), 0.3) if c == 1 else merge_contained(b, 0.6),
        'contain 0.6 + stack text 0.6': lambda b, c: merge_stacked(merge_contained(b, 0.6), 0.6) if c == 1 else merge_contained(b, 0.6),
    }

    print(f"{'conf':>5} {'nms':>4} {'post-processing':32} | {'drawing P/R/F1':>18} | {'text P/R/F1':>18}")
    for conf, nms_iou in configs:
        raw = []
        for f in files:
            r = model.predict(str(f), imgsz=1280, conf=conf, iou=nms_iou, verbose=False)[0]
            page = {0: [], 1: []}
            for box, cls in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                page[int(cls)].append(tuple(box))
            raw.append(page)
        for name, fn in post.items():
            out = []
            for cls in (0, 1):
                preds = [fn(page[cls], cls) for page in raw]
                out.append(prf(*score(preds, [g[cls] for g in gts])))
            (dp, dr, df), (tp_, tr, tf) = out
            print(f"{conf:5.2f} {nms_iou:4.1f} {name:32} | {dp:.2f}/{dr:.2f}/{df:.3f}   | {tp_:.2f}/{tr:.2f}/{tf:.3f}")


if __name__ == '__main__':
    main()
