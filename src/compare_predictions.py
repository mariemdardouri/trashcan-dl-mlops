
from pathlib import Path
import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent

SAMPLE_DIR = ROOT_DIR / "runs" / "visual_analysis" / "sample_images"
GT_DIR = ROOT_DIR / "data" / "processed" / "trashcan_yolo" / "labels" / "val"
PREDICTIONS_DIR = ROOT_DIR / "runs" / "visual_analysis" / "predictions_18"
OUTPUT_DIR = ROOT_DIR / "runs" / "visual_analysis" / "comparisons"

CLASS_NAMES = {
    0: "trash_etc",
    1: "trash_plastic",
    2: "trash_metal",
    3: "trash_wood",
    4: "trash_fabric",
    5: "trash_fishing_gear",
    6: "trash_paper",
    7: "trash_rubber"
}

MODELS = ["baseline", "exp2", "exp3"]


def read_labels(file_path, is_prediction=False):
    boxes = []

    if not file_path.exists():
        return boxes

    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) < 5:
                continue

            class_id = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            confidence = 1.0
            if is_prediction and len(values) >= 6:
                confidence = float(values[5])

            boxes.append({
                "class_id": class_id,
                "x": x_center,
                "y": y_center,
                "w": width,
                "h": height,
                "confidence": confidence
            })

    return boxes


def draw_boxes(image, boxes, color, is_prediction=False):
    height, width = image.shape[:2]

    for box in boxes:
        x_center = box["x"] * width
        y_center = box["y"] * height
        box_width = box["w"] * width
        box_height = box["h"] * height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        class_name = CLASS_NAMES.get(
            box["class_id"], str(box["class_id"])
        )

        label = class_name

        if is_prediction:
            label += f" {box['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text_y = max(y1 - 8, 15)

        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2
        )

    return image


def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_comparison(image_path):
    image_name = image_path.stem

    gt_path = GT_DIR / f"{image_name}.txt"

    ground_truth = read_labels(gt_path)

    panels = []

    # Original image
    original = load_image(image_path)
    panels.append(("Original", original))

    # Ground Truth
    gt_image = load_image(image_path)
    gt_image = draw_boxes(gt_image, ground_truth, (0, 255, 0))
    panels.append(("Ground Truth", gt_image))

    # Model predictions
    for model_name in MODELS:
        pred_path = (
            PREDICTIONS_DIR
            / model_name
            / "labels"
            / f"{image_name}.txt"
        )

        predictions = read_labels(pred_path, is_prediction=True)

        pred_image = load_image(image_path)
        pred_image = draw_boxes(
            pred_image,
            predictions,
            (255, 80, 0),
            is_prediction=True
        )

        panels.append((model_name.upper(), pred_image))

    fig, axes = plt.subplots(1, 5, figsize=(22, 6))

    for axis, panel in zip(axes, panels):
        title, image = panel

        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    fig.suptitle(image_name, fontsize=13)
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{image_name}_comparison.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in SAMPLE_DIR.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    print("=" * 60)
    print("TRASHCAN - VISUAL MODEL COMPARISON")
    print("=" * 60)

    print(f"Images found: {len(image_paths)}")

    successful = 0

    for image_path in image_paths:
        try:
            output_path = create_comparison(image_path)
            successful += 1
            print(f"[OK] {image_path.name}")

        except Exception as error:
            print(f"[ERROR] {image_path.name}: {error}")

    print("\n" + "=" * 60)
    print(f"Comparisons generated: {successful}/{len(image_paths)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
