"""Face detection using Haar cascades (OpenCV) or MediaPipe Tasks API."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

FaceBox = tuple[int, int, int, int]  # x, y, w, h

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_URLS = {
    0: (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ),
    1: (
        "blaze_face_full_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
    ),
}


def haar_cascade_path() -> str:
    return str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")


def detect_faces_haar(
    bgr: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (48, 48),
) -> tuple[list[FaceBox], np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(haar_cascade_path())
    if cascade.empty():
        raise RuntimeError("Haar cascade file missing; reinstall opencv-contrib-python.")
    rects = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]
    return boxes, gray


def _ensure_mediapipe_model(model_selection: int) -> str:
    filename, url = _MODEL_URLS.get(model_selection, _MODEL_URLS[0])
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / filename
    if not path.is_file():
        urllib.request.urlretrieve(url, path)
    return str(path)


class MediaPipeFaceDetector:
    """Wrapper around MediaPipe Tasks FaceDetector (mediapipe >= 0.10.31)."""

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
        for_video: bool = False,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as base_options_module

        model_path = _ensure_mediapipe_model(model_selection)
        running_mode = (
            vision.RunningMode.VIDEO if for_video else vision.RunningMode.IMAGE
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            min_detection_confidence=min_detection_confidence,
        )
        self._mp = mp
        self._for_video = for_video
        self._frame_index = 0
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, bgr: np.ndarray) -> tuple[list[FaceBox], np.ndarray]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if self._for_video:
            timestamp_ms = int(self._frame_index * 1000 / 30)
            self._frame_index += 1
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._detector.detect(mp_image)

        boxes: list[FaceBox] = []
        if result.detections:
            for det in result.detections:
                bbox = det.bounding_box
                x = max(0, int(bbox.origin_x))
                y = max(0, int(bbox.origin_y))
                bw = max(1, int(bbox.width))
                bh = max(1, int(bbox.height))
                boxes.append((x, y, bw, bh))
        return boxes, gray

    def close(self) -> None:
        self._detector.close()


def create_mediapipe_detector(
    min_detection_confidence: float = 0.5,
    model_selection: int = 0,
    *,
    for_video: bool = False,
) -> MediaPipeFaceDetector:
    return MediaPipeFaceDetector(
        min_detection_confidence=min_detection_confidence,
        model_selection=model_selection,
        for_video=for_video,
    )


def detect_faces_mediapipe(
    bgr: np.ndarray,
    min_detection_confidence: float = 0.5,
    model_selection: int = 0,
    detector: MediaPipeFaceDetector | None = None,
) -> tuple[list[FaceBox], np.ndarray, MediaPipeFaceDetector | None]:
    own_detector = detector is None
    det = detector or create_mediapipe_detector(
        min_detection_confidence=min_detection_confidence,
        model_selection=model_selection,
    )
    try:
        boxes, gray = det.detect(bgr)
    finally:
        if own_detector:
            det.close()
    return boxes, gray, (det if not own_detector else None)


def detect_faces(
    bgr: np.ndarray,
    method: str,
    *,
    haar_scale: float = 1.1,
    haar_neighbors: int = 5,
    mp_confidence: float = 0.5,
    mp_model_selection: int = 0,
    mp_detector: MediaPipeFaceDetector | None = None,
) -> tuple[list[FaceBox], np.ndarray, MediaPipeFaceDetector | None]:
    """method: 'haar' | 'mediapipe'. Returns (boxes, gray, mp_detector_or_none)."""
    if method == "haar":
        boxes, gray = detect_faces_haar(
            bgr, scale_factor=haar_scale, min_neighbors=haar_neighbors
        )
        return boxes, gray, None
    if method == "mediapipe":
        return detect_faces_mediapipe(
            bgr,
            min_detection_confidence=mp_confidence,
            model_selection=mp_model_selection,
            detector=mp_detector,
        )
    raise ValueError(f"Unknown detection method: {method}")


def draw_boxes(
    bgr: np.ndarray,
    boxes: list[FaceBox],
    labels: list[str] | None = None,
    color: tuple[int, int, int] = (0, 200, 0),
) -> np.ndarray:
    out = bgr.copy()
    for i, (x, y, bw, bh) in enumerate(boxes):
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(
                out,
                labels[i],
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def crop_face_gray(
    gray: np.ndarray, box: FaceBox, out_size: tuple[int, int] = (200, 200)
) -> np.ndarray:
    x, y, bw, bh = box
    h, w = gray.shape[:2]
    x2, y2 = min(x + bw, w), min(y + bh, h)
    x, y = max(0, x), max(0, y)
    roi = gray[y:y2, x:x2]
    if roi.size == 0:
        return np.zeros(out_size, dtype=np.uint8)
    return cv2.resize(roi, out_size, interpolation=cv2.INTER_AREA)
