"""Streamlit UI: face detection (Haar / MediaPipe) and optional LBPH recognition."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from detection import (
    MediaPipeFaceDetector,
    create_mediapipe_detector,
    crop_face_gray,
    detect_faces,
    draw_boxes,
)
from recognition import predict_face, train_lbph

BASE_DIR = Path(__file__).resolve().parent


def process_image(
    bgr: np.ndarray,
    method: str,
    *,
    use_recognition: bool,
    recognizer,
    names: list[str],
    lbph_threshold: float,
    haar_scale: float,
    haar_neighbors: int,
    mp_confidence: float,
    mp_detector,
) -> np.ndarray:
    boxes, gray, _ = detect_faces(
        bgr,
        method,
        haar_scale=haar_scale,
        haar_neighbors=haar_neighbors,
        mp_confidence=mp_confidence,
        mp_detector=mp_detector,
    )
    labels: list[str] | None = None
    if use_recognition and recognizer is not None and names:
        labels = []
        for box in boxes:
            face = crop_face_gray(gray, box)
            name, _conf = predict_face(recognizer, names, face, lbph_threshold)
            labels.append(name)
    return draw_boxes(bgr, boxes, labels)


def process_video_file(
    path: Path,
    method: str,
    *,
    use_recognition: bool,
    recognizer,
    names: list[str],
    lbph_threshold: float,
    haar_scale: float,
    haar_neighbors: int,
    mp_confidence: float,
    max_frames: int,
    frame_stride: int,
) -> Path | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    import os

    os.close(out_fd)
    writer = cv2.VideoWriter(out_path, fourcc, fps / max(1, frame_stride), (w, h))

    mp_detector: MediaPipeFaceDetector | None = None
    if method == "mediapipe":
        mp_detector = create_mediapipe_detector(
            min_detection_confidence=mp_confidence,
            model_selection=0,
            for_video=True,
        )

    try:
        idx = 0
        written = 0
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride != 0:
                idx += 1
                continue
            boxes, gray, mp_detector = detect_faces(
                frame,
                method,
                haar_scale=haar_scale,
                haar_neighbors=haar_neighbors,
                mp_confidence=mp_confidence,
                mp_detector=mp_detector,
            )
            labels: list[str] | None = None
            if use_recognition and recognizer is not None and names:
                labels = []
                for box in boxes:
                    face = crop_face_gray(gray, box)
                    name, _c = predict_face(recognizer, names, face, lbph_threshold)
                    labels.append(name)
            vis = draw_boxes(frame, boxes, labels)
            writer.write(vis)
            written += 1
            idx += 1
    finally:
        cap.release()
        writer.release()
        if mp_detector is not None:
            mp_detector.close()

    return Path(out_path)


def main() -> None:
    st.set_page_config(
        page_title="Face Detection & Recognition",
        page_icon="🙂",
        layout="wide",
    )
    st.title("Face detection and recognition")
    st.caption(
        "Detect faces with Haar cascades or a MediaPipe neural detector; "
        "optionally recognize enrolled people using OpenCV LBPH."
    )

    with st.sidebar:
        st.header("Detection")
        method = st.radio("Detector", ["mediapipe", "haar"], horizontal=True)
        haar_scale = st.slider("Haar scale factor", 1.01, 1.5, 1.1, 0.01)
        haar_neighbors = st.slider("Haar min neighbors", 3, 12, 5)
        mp_confidence = st.slider("MediaPipe min confidence", 0.1, 0.99, 0.5, 0.01)

        st.header("Recognition (optional)")
        use_recognition = st.checkbox("Label enrolled faces", value=False)
        lbph_threshold = st.slider(
            "LBPH distance threshold (lower = stricter)",
            30.0,
            130.0,
            85.0,
            help="Above this distance, the face is shown as Unknown. Tune after you add samples.",
        )
        st.markdown(
            f"Enroll photos under `{BASE_DIR / 'data' / 'enrolled'}\\<Name>\\` "
            "as `.jpg` / `.png` frontal faces."
        )

    recognizer = None
    names: list[str] = []
    train_msg = ""
    if use_recognition:
        recognizer, names, train_msg = train_lbph(BASE_DIR)
        if train_msg:
            st.info(train_msg)
        if recognizer is None:
            st.warning("Recognition is on but the model could not be trained. Detection still works.")

    tab_img, tab_vid = st.tabs(["Image", "Video"])

    with tab_img:
        up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if up is not None:
            data = np.frombuffer(up.getvalue(), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                st.error("Could not decode image.")
            else:
                mp_det: MediaPipeFaceDetector | None = None
                if method == "mediapipe":
                    mp_det = create_mediapipe_detector(
                        min_detection_confidence=mp_confidence,
                        model_selection=0,
                    )
                try:
                    out = process_image(
                        bgr,
                        method,
                        use_recognition=use_recognition and recognizer is not None,
                        recognizer=recognizer,
                        names=names,
                        lbph_threshold=lbph_threshold,
                        haar_scale=haar_scale,
                        haar_neighbors=haar_neighbors,
                        mp_confidence=mp_confidence,
                        mp_detector=mp_det,
                    )
                finally:
                    if mp_det is not None:
                        mp_det.close()
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)

    with tab_vid:
        st.caption("Processes frames with a stride to keep runs short; output is an annotated MP4.")
        vup = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
        max_frames = st.number_input("Max frames to write", 30, 600, 120, 10)
        frame_stride = st.number_input("Frame stride (1 = every frame)", 1, 10, 2)
        if vup is not None:
            suffix = Path(vup.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(vup.getbuffer())
                tmp_path = Path(tmp.name)
            try:
                with st.spinner("Processing video…"):
                    outp = process_video_file(
                        tmp_path,
                        method,
                        use_recognition=use_recognition and recognizer is not None,
                        recognizer=recognizer,
                        names=names,
                        lbph_threshold=lbph_threshold,
                        haar_scale=haar_scale,
                        haar_neighbors=haar_neighbors,
                        mp_confidence=mp_confidence,
                        max_frames=int(max_frames),
                        frame_stride=int(frame_stride),
                    )
                if outp is None:
                    st.error("Could not open video.")
                else:
                    st.success("Done.")
                    st.video(str(outp))
                    with open(outp, "rb") as f:
                        st.download_button(
                            "Download annotated video",
                            f.read(),
                            file_name="faces_annotated.mp4",
                            mime="video/mp4",
                        )
            finally:
                tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
