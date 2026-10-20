"""Fine-tune a COCO-pretrained YOLO on the dataset made by prepare_dataset.py.

Usage:
    pip install ultralytics
    python detector/prepare_dataset.py --holdout Stromboli_1
    python detector/train.py
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolo11s.pt')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--device', default=None, help="e.g. 0 for the first GPU, or cpu")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=str(HERE / 'dataset' / 'data.yaml'),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(HERE / 'runs'),
        name='scan_detector',
        # scans are always upright after exif_transpose and pages are not mirrored:
        # keep geometric augmentation mild, no flips
        fliplr=0.0,
        flipud=0.0,
        degrees=3.0,
        mosaic=0.5,
        patience=30,
    )


if __name__ == '__main__':
    main()
