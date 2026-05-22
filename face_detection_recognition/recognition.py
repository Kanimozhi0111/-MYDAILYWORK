"""Optional face recognition using OpenCV LBPH on enrolled face crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detection import FaceBox, crop_face_gray, detect_faces_haar

OUT_SIZE = (200, 200)


def enrolled_root(base: Path) -> Path:
    return base / "data" / "enrolled"


def list_people(base: Path) -> list[str]:
    root = enrolled_root(base)
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _load_training_faces(base: Path) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Return (faces, labels_as_int, label_names_ordered_by_id)."""
    people = list_people(base)
    faces: list[np.ndarray] = []
    labels: list[int] = []
    for idx, name in enumerate(people):
        folder = enrolled_root(base) / name
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            boxes, gray = detect_faces_haar(img)
            if not boxes:
                continue
            x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
            face = crop_face_gray(gray, (x, y, bw, bh), OUT_SIZE)
            faces.append(face)
            labels.append(idx)
    label_names = people
    return faces, labels, label_names


def train_lbph(base: Path) -> tuple[Any, list[str], str]:
    """Train LBPH on ``data/enrolled/<Person>/*.jpg``. Returns (recognizer, names, status_message)."""
    faces, labels, names = _load_training_faces(base)
    if not names:
        return None, [], "No enrolled people found. Add folders under data/enrolled/."
    if len(faces) == 0:
        return (
            None,
            names,
            "Enrolled folders exist but no usable face crops were found. Add clear frontal photos.",
        )
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    return recognizer, names, f"Trained LBPH on {len(faces)} face sample(s) across {len(names)} person(s)."


def predict_face(
    recognizer: Any,
    names: list[str],
    gray_face: np.ndarray,
    confidence_threshold: float = 95.0,
) -> tuple[str, float]:
    """
    LBPH: lower confidence is a better match. OpenCV uses a distance; treat above threshold as unknown.
    """
    label_id, confidence = recognizer.predict(gray_face)
    if label_id < 0 or label_id >= len(names):
        return "Unknown", float(confidence)
    if confidence > confidence_threshold:
        return "Unknown", float(confidence)
    return names[label_id], float(confidence)
