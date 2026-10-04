"""Drawing / text-box detector (YOLO) used by the "Auto-detect" button.

The model finds two classes on a scanned plate: 0 = drawing, 1 = text. Each text box is then
attached to its nearest drawing, giving the same {x, y, w, h, textBoxes: []} structure the
annotation tab saves by hand. Coordinates are in the EXIF-transposed image, i.e. the frame
the browser shows and the annotations are stored in.
"""
import logging
import os
import threading

from PIL import Image, ImageOps

from app.config import Config

logger = logging.getLogger(__name__)

DRAWING, TEXT = 0, 1

# Tuned on the validation pages (detector/tune_postprocess.py): F1 drawings 0.905 -> 0.935,
# text 0.839 -> 0.871 over ultralytics' defaults (conf 0.25, iou 0.7, no merging).
CONF = 0.4
NMS_IOU = 0.5
CONTAINED_MERGE = 0.8


class DetectorUnavailable(RuntimeError):
    """The detector weights are missing and could not be downloaded."""


class Detector:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    @property
    def weights_path(self):
        return os.path.join(Config.DETECTOR_MODEL_DIR, Config.DETECTOR_MODEL_FILE)

    def _ensure_weights(self):
        if os.path.exists(self.weights_path):
            return
        logger.info(f"📥 Detector weights not found, downloading {Config.DETECTOR_MODEL_ID}...")
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs(Config.DETECTOR_MODEL_DIR, exist_ok=True)
            hf_hub_download(
                Config.DETECTOR_MODEL_ID,
                Config.DETECTOR_MODEL_FILE,
                local_dir=Config.DETECTOR_MODEL_DIR,
            )
        except Exception as e:
            raise DetectorUnavailable(
                f"Detector weights not found at {self.weights_path} and download from "
                f"{Config.DETECTOR_MODEL_ID} failed: {e}"
            )

    def _load(self):
        if self._model is None:
            self._ensure_weights()
            try:
                from ultralytics import YOLO
            except ImportError:
                raise DetectorUnavailable("The 'ultralytics' package is not installed (pip install ultralytics)")
            self._model = YOLO(self.weights_path)
            logger.info("✅ Detector loaded")
        return self._model

    def detect(self, image_path, conf=CONF, imgsz=1280):
        """Return [{x, y, w, h, textBoxes: [{x, y, w, h}]}] for one image (rounded pixel ints)."""
        with self._lock:
            model = self._load()
            image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
            result = model.predict(image, imgsz=imgsz, conf=conf, iou=NMS_IOU, verbose=False)[0]

        found = {DRAWING: [], TEXT: []}
        for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
            found[int(cls)].append(tuple(box))
        drawings, texts = (
            [{'x': round(x0), 'y': round(y0), 'w': round(x1 - x0), 'h': round(y1 - y0)}
             for x0, y0, x1, y1 in _merge_contained(found[cls])]
            for cls in (DRAWING, TEXT)
        )

        for d in drawings:
            d['textBoxes'] = []
        for t in texts:
            if drawings:
                min(drawings, key=lambda d: _label_distance(d, t))['textBoxes'].append(t)

        # Reading order: top-to-bottom, then left-to-right
        drawings.sort(key=lambda d: (d['y'], d['x']))
        for d in drawings:
            d['textBoxes'].sort(key=lambda t: (t['y'], t['x']))
        return drawings


def _merge_contained(boxes, thr=CONTAINED_MERGE):
    """Merge same-class boxes when the overlap covers >= thr of the smaller one.

    YOLO's NMS only looks at IoU, so a small box lying inside a larger one (one label split in
    two) survives it. Repeats until no pair is left."""
    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                overlap = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
                if overlap >= thr * min(area(a), area(b)):
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _label_distance(drawing, text):
    """How far a text box is from a drawing it might label.

    Labels sit beside or above their drawing far more often than below it, so a drawing that
    lies above the text counts as 3x further away. On the annotated projects this raised
    the right text->drawing match from 84.2% to 86.5% over plain distance.
    """
    d = _gap(drawing, text)
    return d * 3 if text['y'] >= drawing['y'] + drawing['h'] * 0.5 else d


def _gap(a, b):
    """Distance between two rectangles (0 if they overlap)."""
    dx = max(a['x'] - (b['x'] + b['w']), b['x'] - (a['x'] + a['w']), 0)
    dy = max(a['y'] - (b['y'] + b['h']), b['y'] - (a['y'] + a['h']), 0)
    return (dx * dx + dy * dy) ** 0.5


detector = Detector()
