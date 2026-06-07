from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    import torch
except Exception as exc:  # pragma: no cover - shown in Streamlit UI.
    cv2 = None
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parent
YOLOV7_DIR = ROOT / "yolov7"
IMAGES_DIR = ROOT / "images"
SAMPLE_SUBMISSION = ROOT / "sample_submission.csv"
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    weights: Path
    postprocess: str
    default_confidence: float
    color: tuple[int, int, int]


MODEL_SPECS: dict[str, ModelSpec] = {
    "Implementation 01 - WBF": ModelSpec(
        label="Implementation 01 - WBF",
        weights=ROOT / "01_best.pt",
        postprocess="wbf",
        default_confidence=0.60,
        color=(34, 139, 230),
    ),
    "Implementation 08 - NMS": ModelSpec(
        label="Implementation 08 - NMS",
        weights=ROOT / "08_best.pt",
        postprocess="nms",
        default_confidence=0.70,
        color=(240, 101, 67),
    ),
}


@dataclass(frozen=True)
class Detection:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float
    class_id: int

    @property
    def area(self) -> int:
        return max(0, self.x_max - self.x_min) * max(0, self.y_max - self.y_min)


def configure_page() -> None:
    st.set_page_config(
        page_title="YOLOv7 Detection Demo",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; }
        .stDataFrame { border: 1px solid rgba(49, 51, 63, 0.16); border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def list_images() -> list[Path]:
    if not IMAGES_DIR.exists():
        return []
    return sorted(
        path
        for path in IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES
    )


def read_sample_submission() -> pd.DataFrame | None:
    if not SAMPLE_SUBMISSION.exists():
        return None
    try:
        return pd.read_csv(SAMPLE_SUBMISSION)
    except Exception:
        return None


def check_artifacts() -> list[tuple[str, bool]]:
    return [
        ("01_best.pt", (ROOT / "01_best.pt").exists()),
        ("08_best.pt", (ROOT / "08_best.pt").exists()),
        ("v3_01_WBF_outputcsv.ipynb", (ROOT / "v3_01_WBF_outputcsv.ipynb").exists()),
        ("v5_08_NMS_outputcsv.ipynb", (ROOT / "v5_08_NMS_outputcsv.ipynb").exists()),
        ("sample_submission.csv", SAMPLE_SUBMISSION.exists()),
        ("images/", IMAGES_DIR.exists()),
        ("yolov7/", YOLOV7_DIR.exists()),
        ("ensemble-boxes", importlib.util.find_spec("ensemble_boxes") is not None),
    ]


def add_yolov7_to_path() -> None:
    yolov7_path = str(YOLOV7_DIR)
    if yolov7_path not in sys.path:
        sys.path.insert(0, yolov7_path)


def install_cbam_into_yolov7() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required.")

    import torch.nn as nn
    import models.common as common

    if hasattr(common, "CBAM"):
        return

    class ChannelAttention(nn.Module):
        def __init__(self, in_planes: int, ratio: int = 16) -> None:
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
            self.relu = nn.ReLU()
            self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
            max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
            return self.sigmoid(avg_out + max_out)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size: int = 7) -> None:
            super().__init__()
            if kernel_size not in (3, 7):
                raise ValueError("kernel_size must be 3 or 7")
            padding = 3 if kernel_size == 7 else 1
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            return self.sigmoid(self.conv(x))

    class CBAM(nn.Module):
        def __init__(self, c1: int, ratio: int = 16, kernel_size: int = 7) -> None:
            super().__init__()
            self.channel_attention = ChannelAttention(c1, ratio)
            self.spatial_attention = SpatialAttention(kernel_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.channel_attention(x) * x
            return self.spatial_attention(x) * x

    common.ChannelAttention = ChannelAttention
    common.SpatialAttention = SpatialAttention
    common.CBAM = CBAM


@contextlib.contextmanager
def force_full_torch_load() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required.")

    if "weights_only" not in inspect.signature(torch.load).parameters:
        yield
        return

    original_torch_load = torch.load

    def patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


@st.cache_resource(show_spinner=False)
def load_yolov7_model(weights_path: str, device_name: str) -> tuple[Any, int]:
    if IMPORT_ERROR is not None:
        raise RuntimeError(f"Missing Python dependency: {IMPORT_ERROR}")
    if not YOLOV7_DIR.exists():
        raise FileNotFoundError("Missing local yolov7/ repository.")

    add_yolov7_to_path()
    install_cbam_into_yolov7()

    from models.experimental import attempt_load

    device = torch.device(device_name)
    with force_full_torch_load():
        model = attempt_load(weights_path, map_location=device)
    model.eval()
    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    return model, stride


@st.cache_data(show_spinner=False)
def load_image_from_disk(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def prepare_image_tensor(
    image: Image.Image,
    img_size: int,
    stride: int,
    device_name: str,
) -> tuple[np.ndarray, Any, tuple[int, int]]:
    add_yolov7_to_path()
    from utils.datasets import letterbox

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized = letterbox(image_bgr, img_size, stride=stride, auto=False)[0]
    resized = resized[:, :, ::-1].transpose(2, 0, 1)
    resized = np.ascontiguousarray(resized)

    tensor = torch.from_numpy(resized).to(torch.device(device_name)).float() / 255.0
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)
    return image_bgr, tensor, tensor.shape[2:]


def wbf_postprocess(
    prediction: Any,
    conf_thres: float,
    iou_thres: float,
    img_shape: tuple[int, int],
    target_class: int,
) -> list[Any]:
    from ensemble_boxes import weighted_boxes_fusion
    from utils.general import xywh2xyxy

    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for batch_index, batch_prediction in enumerate(prediction):
        filtered = batch_prediction[batch_prediction[:, 4] > conf_thres]
        if not filtered.shape[0]:
            continue

        filtered[:, 5:] *= filtered[:, 4:5]
        conf, class_ids = filtered[:, 5:].max(1, keepdim=True)
        filtered = torch.cat(
            (xywh2xyxy(filtered[:, :4]), conf, class_ids.float()),
            1,
        )[conf.view(-1) > conf_thres]
        if not filtered.shape[0]:
            continue

        filtered = filtered[filtered[:, 5] == target_class]
        if not filtered.shape[0]:
            continue

        boxes = filtered[:, :4].detach().cpu().numpy()
        boxes[:, [0, 2]] /= img_shape[1]
        boxes[:, [1, 3]] /= img_shape[0]
        boxes = np.clip(boxes, 0.0, 1.0)

        scores = filtered[:, 4].detach().cpu().numpy()
        labels = filtered[:, 5].detach().cpu().numpy()
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            [boxes.tolist()],
            [scores.tolist()],
            [labels.tolist()],
            weights=None,
            iou_thr=iou_thres,
            skip_box_thr=0.0,
        )

        if len(fused_boxes) == 0:
            continue

        fused_boxes = torch.tensor(fused_boxes, device=prediction.device)
        fused_boxes[:, [0, 2]] *= img_shape[1]
        fused_boxes[:, [1, 3]] *= img_shape[0]
        fused_scores = torch.tensor(fused_scores, device=prediction.device).unsqueeze(1)
        fused_labels = torch.tensor(fused_labels, device=prediction.device).unsqueeze(1)
        outputs[batch_index] = torch.cat((fused_boxes, fused_scores, fused_labels), 1)

    return outputs


def standard_nms_postprocess(
    prediction: Any,
    conf_thres: float,
    iou_thres: float,
    target_class: int,
) -> list[Any]:
    from utils.general import non_max_suppression

    return non_max_suppression(
        prediction,
        conf_thres,
        iou_thres,
        classes=[target_class],
    )


def to_detections(
    processed_predictions: list[Any],
    input_shape: tuple[int, int],
    original_bgr: np.ndarray,
    min_area: int,
) -> tuple[list[Detection], int]:
    from utils.general import scale_coords

    height, width = original_bgr.shape[:2]
    detections: list[Detection] = []
    removed = 0

    for det in processed_predictions:
        if len(det) == 0:
            continue

        det[:, :4] = scale_coords(input_shape, det[:, :4], original_bgr.shape).round()
        for row in det:
            x_min, y_min, x_max, y_max = [int(value) for value in row[:4]]
            x_min = min(max(x_min, 0), width)
            y_min = min(max(y_min, 0), height)
            x_max = min(max(x_max, 0), width)
            y_max = min(max(y_max, 0), height)
            score = float(row[4])
            class_id = int(row[5])

            detection = Detection(x_min, y_min, x_max, y_max, score, class_id)
            if detection.area < min_area:
                removed += 1
                continue
            detections.append(detection)

    detections.sort(key=lambda item: item.score, reverse=True)
    return detections, removed


def run_detection(
    spec: ModelSpec,
    image: Image.Image,
    confidence: float,
    iou: float,
    image_size: int,
    target_class: int,
    min_area: int,
    use_tta: bool,
    device_name: str,
) -> tuple[list[Detection], int]:
    model, stride = load_yolov7_model(str(spec.weights), device_name)
    original_bgr, tensor, input_shape = prepare_image_tensor(
        image,
        image_size,
        stride,
        device_name,
    )

    with torch.inference_mode():
        prediction = model(tensor, augment=use_tta)[0]

    if spec.postprocess == "wbf":
        processed = wbf_postprocess(
            prediction.clone(),
            conf_thres=confidence,
            iou_thres=iou,
            img_shape=input_shape,
            target_class=target_class,
        )
    else:
        processed = standard_nms_postprocess(
            prediction.clone(),
            conf_thres=confidence,
            iou_thres=iou,
            target_class=target_class,
        )

    return to_detections(processed, input_shape, original_bgr, min_area)


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    color: tuple[int, int, int],
    max_draw: int,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in detections[:max_draw]:
        box = [detection.x_min, detection.y_min, detection.x_max, detection.y_max]
        draw.rectangle(box, outline=color, width=3)
        label = f"{detection.score:.2f}"
        label_box = draw.textbbox((box[0], box[1]), label, font=font)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 6
        label_y = max(0, box[1] - label_height)
        draw.rectangle(
            [box[0], label_y, box[0] + label_width, label_y + label_height],
            fill=color,
        )
        draw.text((box[0] + 4, label_y + 3), label, fill=(255, 255, 255), font=font)

    return annotated


def detections_to_frame(detections: list[Detection]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "x_min": item.x_min,
                "y_min": item.y_min,
                "x_max": item.x_max,
                "y_max": item.y_max,
                "score": round(item.score, 4),
                "class_id": item.class_id,
                "area": item.area,
            }
            for item in detections
        ]
    )


def kaggle_bbox_string(detections: list[Detection]) -> str:
    if not detections:
        return "0"
    return " ".join(
        f"{item.x_min} {item.y_min} {item.x_max} {item.y_max}" for item in detections
    )


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_artifact_status() -> None:
    statuses = check_artifacts()
    missing = [name for name, exists in statuses if not exists]
    cols = st.columns(4)
    image_count = len(list_images())
    sample = read_sample_submission()
    sample_rows = len(sample) if sample is not None else 0
    model_count = sum(1 for name, exists in statuses if name.endswith(".pt") and exists)

    cols[0].metric("Models", f"{model_count}/2")
    cols[1].metric("Images", image_count)
    cols[2].metric("Sample Rows", sample_rows)
    cols[3].metric("Missing", len(missing))

    if missing:
        st.warning("Missing required files: " + ", ".join(missing))


def select_image_from_sidebar() -> tuple[Image.Image | None, str]:
    image_files = list_images()
    source = st.sidebar.radio("Image source", ["images folder", "upload"], horizontal=True)
    if source == "upload":
        uploaded = st.sidebar.file_uploader(
            "Upload image",
            type=sorted(ext.lstrip(".") for ext in VALID_IMAGE_SUFFIXES),
        )
        if uploaded is None:
            return None, "uploaded_image"
        return Image.open(uploaded).convert("RGB"), uploaded.name

    if not image_files:
        return None, "selected_image"

    selected = st.sidebar.selectbox(
        "Image",
        image_files,
        format_func=lambda path: path.name,
    )
    return load_image_from_disk(str(selected)), selected.name


def main() -> None:
    configure_page()

    st.title("YOLOv7 車輛偵測成果展示")
    st.caption("讀取本地模型與測試影像，輸出偵測框、統計與 Kaggle bbox 格式。")

    render_artifact_status()

    st.sidebar.header("Inference")
    selected_model_key = st.sidebar.selectbox("Model", list(MODEL_SPECS))
    spec = MODEL_SPECS[selected_model_key]
    confidence = st.sidebar.slider(
        "Confidence",
        min_value=0.05,
        max_value=0.95,
        value=spec.default_confidence,
        step=0.05,
    )
    iou = st.sidebar.slider("IoU", 0.05, 0.95, 0.35, 0.05)
    image_size = st.sidebar.select_slider(
        "Image size",
        options=[640, 960, 1280, 1600, 1920, 2560, 3200],
        value=3200,
    )
    target_class = st.sidebar.number_input("Target class", min_value=0, value=2, step=1)
    min_area = st.sidebar.number_input("Min area", min_value=0, value=80, step=10)
    use_tta = st.sidebar.checkbox("TTA", value=True)
    max_draw = st.sidebar.slider("Max boxes shown", 10, 2000, 600, 10)
    device_options = ["cuda:0", "cpu"] if torch is not None and torch.cuda.is_available() else ["cpu"]
    device_name = st.sidebar.selectbox("Device", device_options)

    image, image_name = select_image_from_sidebar()
    if image is None:
        st.info("Select or upload an image.")
        return

    if IMPORT_ERROR is not None:
        st.error(f"Missing dependency: {IMPORT_ERROR}")
        return
    if not spec.weights.exists():
        st.error(f"Missing model weights: {spec.weights.name}")
        return
    if spec.postprocess == "wbf" and importlib.util.find_spec("ensemble_boxes") is None:
        st.error("Missing `ensemble-boxes`. Install it with `pip install -r requirements_streamlit.txt`.")
        st.image(image, caption=image_name, use_column_width=True)
        return
    if not YOLOV7_DIR.exists():
        st.error("Missing `yolov7/`. See `STREAMLIT_DEMO.md` for setup.")
        st.image(image, caption=image_name, use_column_width=True)
        return

    action_cols = st.columns([0.24, 0.76])
    run_main = action_cols[0].button(
        "Run detection",
        type="primary",
        use_container_width=True,
        key="run_detection_main",
    )
    action_cols[1].caption(
        "設定參數後按 Run detection；未執行前右側只會顯示 preview，不會有框。"
    )

    left, right = st.columns([1.05, 0.95], gap="large")
    left.image(image, caption=f"Original: {image_name}", use_column_width=True)

    run_sidebar = st.sidebar.button(
        "Run detection",
        type="primary",
        use_container_width=True,
        key="run_detection_sidebar",
    )
    run = run_main or run_sidebar
    if not run:
        right.image(image, caption="Detection preview", use_column_width=True)
        st.info("目前尚未執行推論。按上方或側欄的 Run detection 後才會畫出偵測框。")
        return

    with st.spinner("Running YOLOv7 inference..."):
        detections, removed = run_detection(
            spec=spec,
            image=image,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
            target_class=int(target_class),
            min_area=int(min_area),
            use_tta=use_tta,
            device_name=device_name,
        )
        annotated = draw_detections(image, detections, spec.color, max_draw)

    right.image(
        annotated,
        caption=f"{spec.label}: {len(detections)} boxes",
        use_column_width=True,
    )

    stat_cols = st.columns(4)
    stat_cols[0].metric("Detected", len(detections))
    stat_cols[1].metric("Filtered", removed)
    stat_cols[2].metric("Confidence", f"{confidence:.2f}")
    stat_cols[3].metric("Postprocess", spec.postprocess.upper())

    frame = detections_to_frame(detections)
    st.subheader("Detection Boxes")
    st.dataframe(frame, use_container_width=True, hide_index=True)

    bbox_row = pd.DataFrame([{"ID": image_name, "bbox": kaggle_bbox_string(detections)}])
    csv_bytes = bbox_row.to_csv(index=False).encode("utf-8")
    dl_cols = st.columns(2)
    dl_cols[0].download_button(
        "Download annotated PNG",
        image_to_png_bytes(annotated),
        file_name=f"{Path(image_name).stem}_{spec.postprocess}_annotated.png",
        mime="image/png",
        use_container_width=True,
    )
    dl_cols[1].download_button(
        "Download bbox CSV row",
        csv_bytes,
        file_name=f"{Path(image_name).stem}_{spec.postprocess}_bbox.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
