
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Annotation Checker",
    page_icon="🔍",
    layout="wide"
)

# ─── Title & Description ─────────────────────────────────────────────────────────
st.title("🔍 Image Annotation Checker")
st.markdown("Upload an image and its YOLO annotation file to visualize and validate bounding boxes.")
st.divider()

# ─── Color Palette for Different Classes ─────────────────────────────────────────
# Each class gets a distinct color (BGR format for OpenCV)
CLASS_COLORS = [
    (255,  56,  56),   # Red
    ( 56, 255,  56),   # Green
    ( 56,  56, 255),   # Blue
    (255, 200,  56),   # Yellow
    (200,  56, 255),   # Purple
    ( 56, 200, 255),   # Cyan
    (255, 128,   0),   # Orange
    (  0, 200, 128),   # Teal
]

def get_color(class_id: int) -> tuple:
    """Return a consistent color for a given class ID."""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def parse_annotations(annotation_text: str) -> list[dict] | str:
    """
    Parse YOLO annotation text into a list of annotation dictionaries.

    Each line in a YOLO .txt file looks like:
        class_id  x_center  y_center  width  height
    All values except class_id are floats between 0 and 1.

    Returns a list of dicts on success, or an error string on failure.
    """
    annotations = []
    lines = annotation_text.strip().splitlines()

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:           # skip blank lines
            continue

        parts = line.split()
        if len(parts) != 5:
            return f"Line {line_num}: expected 5 values, got {len(parts)} → '{line}'"

        try:
            class_id                          = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            return f"Line {line_num}: could not parse numbers → '{line}'"

        annotations.append({
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width":    width,
            "height":   height,
            "line_num": line_num,
        })

    return annotations


def yolo_to_pixel(ann: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """
    Convert YOLO normalized coordinates to pixel coordinates.

    YOLO stores box centres and dimensions as fractions of the image size.
    We multiply by the actual pixel dimensions to get real coordinates.

    Returns (x_min, y_min, x_max, y_max) in pixels.
    """
    x_center_px = ann["x_center"] * img_w
    y_center_px = ann["y_center"] * img_h
    box_w_px    = ann["width"]    * img_w
    box_h_px    = ann["height"]   * img_h

    x_min = int(x_center_px - box_w_px / 2)
    y_min = int(y_center_px - box_h_px / 2)
    x_max = int(x_center_px + box_w_px / 2)
    y_max = int(y_center_px + box_h_px / 2)

    return x_min, y_min, x_max, y_max


def validate_annotation(ann: dict) -> list[str]:
    """
    Check whether YOLO values are within legal range (0–1).

    Returns a list of problem strings (empty list = annotation is valid).
    """
    issues = []
    fields = {
        "x_center": ann["x_center"],
        "y_center": ann["y_center"],
        "width":    ann["width"],
        "height":   ann["height"],
    }
    for name, val in fields.items():
        if not (0.0 <= val <= 1.0):
            issues.append(f"{name}={val:.4f} is outside [0, 1]")

    # Bounding box must not extend outside the image
    half_w = ann["width"]  / 2
    half_h = ann["height"] / 2
    if ann["x_center"] - half_w < 0 or ann["x_center"] + half_w > 1:
        issues.append("Box extends outside image horizontally")
    if ann["y_center"] - half_h < 0 or ann["y_center"] + half_h > 1:
        issues.append("Box extends outside image vertically")

    return issues


def draw_annotations(image_np: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """
    Draw bounding boxes and class labels on a copy of the image.

    Returns:
        annotated_image  – NumPy array with boxes drawn
        results          – list of per-annotation validation info
    """
    img_h, img_w = image_np.shape[:2]
    output       = image_np.copy()
    results      = []

    for ann in annotations:
        issues           = validate_annotation(ann)
        is_valid         = len(issues) == 0
        x_min, y_min, x_max, y_max = yolo_to_pixel(ann, img_w, img_h)

        # Clamp to image boundaries so drawing doesn't fail
        x_min_draw = max(0, x_min)
        y_min_draw = max(0, y_min)
        x_max_draw = min(img_w, x_max)
        y_max_draw = min(img_h, y_max)

        color     = get_color(ann["class_id"]) if is_valid else (0, 0, 200)
        thickness = 2

        # Draw the rectangle
        cv2.rectangle(output,
                      (x_min_draw, y_min_draw),
                      (x_max_draw, y_max_draw),
                      color, thickness)

        # Build label text
        label       = f"Class {ann['class_id']}"
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = max(0.4, min(img_w, img_h) / 800)
        text_thick  = 1

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thick)

        # Label background rectangle
        lx1 = x_min_draw
        ly1 = max(0, y_min_draw - th - baseline - 4)
        lx2 = x_min_draw + tw + 6
        ly2 = y_min_draw

        cv2.rectangle(output, (lx1, ly1), (lx2, ly2), color, -1)   # filled

        # Label text (black for readability on coloured background)
        cv2.putText(output, label,
                    (lx1 + 3, ly2 - baseline - 1),
                    font, font_scale, (0, 0, 0), text_thick,
                    cv2.LINE_AA)

        results.append({
            "line_num":  ann["line_num"],
            "class_id":  ann["class_id"],
            "is_valid":  is_valid,
            "issues":    issues,
            "pixel_box": (x_min, y_min, x_max, y_max),
        })

    return output, results


# ─── Sidebar: Upload Widgets ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Files")

    uploaded_image = st.file_uploader(
        "1. Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, PNG, BMP, WEBP"
    )

    uploaded_annotation = st.file_uploader(
        "2. Upload YOLO Annotation (.txt)",
        type=["txt"],
        help="One annotation per line: class_id x_center y_center width height"
    )

    st.divider()
    st.markdown("**YOLO Format reminder:**")
    st.code("class_id  x_center  y_center  width  height", language="text")
    st.caption("All coordinates are normalised (0–1).")

# ─── Main Content ─────────────────────────────────────────────────────────────────
if uploaded_image is None or uploaded_annotation is None:
    # Welcome / instruction screen
    col1, col2 = st.columns(2)
    with col1:
        st.info("👈 Upload an **image** using the sidebar to get started.")
    with col2:
        st.info("👈 Upload a **YOLO .txt** annotation file too.")

    st.markdown("### 📖 How it works")
    steps = [
        ("1️⃣ Upload Image", "Any common image format (JPG, PNG, …)."),
        ("2️⃣ Upload Annotation", "A YOLO `.txt` file with one box per line."),
        ("3️⃣ View Results",  "Bounding boxes are drawn; each annotation is validated."),
    ]
    cols = st.columns(3)
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.write(desc)

else:
    # ── Load image ──────────────────────────────────────────────────────────────
    pil_image  = Image.open(uploaded_image).convert("RGB")
    image_np   = np.array(pil_image)             # RGB numpy array
    image_bgr  = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # BGR for OpenCV

    img_h, img_w = image_np.shape[:2]

    # ── Parse annotation file ───────────────────────────────────────────────────
    annotation_text = uploaded_annotation.read().decode("utf-8")
    parse_result    = parse_annotations(annotation_text)

    if isinstance(parse_result, str):
        # parse_result is an error message string
        st.error(f"❌ **Annotation parse error:** {parse_result}")
        st.stop()

    annotations = parse_result   # list of dicts

    if len(annotations) == 0:
        st.warning("⚠️ The annotation file is empty — nothing to draw.")
        st.image(pil_image, caption="Uploaded Image (no annotations)", use_container_width=True)
        st.stop()

    # ── Draw & validate ─────────────────────────────────────────────────────────
    annotated_bgr, results = draw_annotations(image_bgr, annotations)
    annotated_rgb          = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    valid_count   = sum(1 for r in results if r["is_valid"])
    invalid_count = len(results) - valid_count

    # ── Summary metrics ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🖼️ Image size",        f"{img_w} × {img_h} px")
    c2.metric("📝 Total annotations", len(annotations))
    c3.metric("✅ Valid",              valid_count)
    c4.metric("❌ Invalid",            invalid_count)

    st.divider()

    # ── Image columns ────────────────────────────────────────────────────────────
    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.subheader("Original Image")
        st.image(pil_image, use_container_width=True)

    with col_ann:
        st.subheader("Annotated Image")
        st.image(annotated_rgb, use_container_width=True)

    # ── Validation banner ────────────────────────────────────────────────────────
    st.divider()
    if invalid_count == 0:
        st.success(f"✅ All {len(annotations)} annotation(s) are **valid**!")
    else:
        st.error(f"❌ {invalid_count} of {len(annotations)} annotation(s) have issues.")

    # ── Per-annotation detail table ──────────────────────────────────────────────
    st.subheader("📋 Annotation Details")

    for r in results:
        x1, y1, x2, y2 = r["pixel_box"]
        status_icon = "✅" if r["is_valid"] else "❌"
        status_text = "Valid" if r["is_valid"] else "Invalid"

        with st.expander(
            f"{status_icon} Line {r['line_num']} — Class {r['class_id']} — {status_text}",
            expanded=not r["is_valid"]   # auto-expand invalid ones
        ):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Class ID:** `{r['class_id']}`")
                st.markdown(f"**Status:** {status_text}")
            with col_b:
                st.markdown(f"**Pixel box:** `({x1}, {y1}) → ({x2}, {y2})`")

            if r["issues"]:
                for issue in r["issues"]:
                    st.warning(f"⚠️ {issue}")

    # ── Download button ───────────────────────────────────────────────────────────
    st.divider()
    annotated_pil = Image.fromarray(annotated_rgb)
    buf           = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        label     = "⬇️ Download Annotated Image",
        data      = buf.getvalue(),
        file_name = "annotated_result.png",
        mime      = "image/png",
    )
