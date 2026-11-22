
from pathlib import Path
import csv

ROOT_DIR = Path(__file__).resolve().parent.parent

SAMPLE_DIR = ROOT_DIR / "runs" / "visual_analysis" / "sample_images"
GT_DIR = ROOT_DIR / "data" / "processed" / "trashcan_yolo" / "labels" / "val"
PREDICTIONS_DIR = ROOT_DIR / "runs" / "visual_analysis" / "predictions_18"

MODELS = ["baseline", "exp2", "exp3"]
IOU_THRESHOLD = 0.50

CLASS_NAMES = {
    0: "trash_etc",
    1: "trash_plastic",
    2: "trash_metal",
    3: "trash_wood",
    4: "trash_fabric",
    5: "trash_fishing_gear",
    6: "trash_paper",
    7: "trash_rubber",
}


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

            confidence = float(values[5]) if is_prediction and len(values) >= 6 else 1.0

            boxes.append({
                "class_id": class_id,
                "x": x_center,
                "y": y_center,
                "w": width,
                "h": height,
                "confidence": confidence
            })

    return boxes


def calculate_iou(box1, box2):
    x1_min = box1["x"] - box1["w"] / 2
    y1_min = box1["y"] - box1["h"] / 2
    x1_max = box1["x"] + box1["w"] / 2
    y1_max = box1["y"] + box1["h"] / 2

    x2_min = box2["x"] - box2["w"] / 2
    y2_min = box2["y"] - box2["h"] / 2
    x2_max = box2["x"] + box2["w"] / 2
    y2_max = box2["y"] + box2["h"] / 2

    intersection_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    intersection_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    intersection = intersection_x * intersection_y

    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def evaluate_model(model_name):
    prediction_dir = PREDICTIONS_DIR / model_name / "labels"

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_path in sorted(SAMPLE_DIR.iterdir()):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image_name = image_path.stem

        gt_path = GT_DIR / f"{image_name}.txt"
        pred_path = prediction_dir / f"{image_name}.txt"

        ground_truth = read_labels(gt_path)
        predictions = read_labels(pred_path, is_prediction=True)

        # Highest-confidence predictions are matched first.
        predictions.sort(
            key=lambda item: item["confidence"],
            reverse=True
        )

        matched_gt = set()

        for prediction in predictions:
            best_iou = 0.0
            best_gt_index = None

            for gt_index, gt in enumerate(ground_truth):
                if gt_index in matched_gt:
                    continue

                if prediction["class_id"] != gt["class_id"]:
                    continue

                iou = calculate_iou(prediction, gt)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_gt_index is not None and best_iou >= IOU_THRESHOLD:
                total_tp += 1
                matched_gt.add(best_gt_index)
            else:
                total_fp += 1

        total_fn += len(ground_truth) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {
        "model": model_name,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


def main():
    results = []

    print("\nVISUAL SAMPLE EVALUATION")
    print("=" * 65)
    print(f"Images: {SAMPLE_DIR}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print("=" * 65)

    for model_name in MODELS:
        result = evaluate_model(model_name)
        results.append(result)

        print(f"\n{model_name.upper()}")
        print(f"TP: {result['TP']}")
        print(f"FP: {result['FP']}")
        print(f"FN: {result['FN']}")
        print(f"Precision: {result['Precision']:.4f}")
        print(f"Recall: {result['Recall']:.4f}")
        print(f"F1-score: {result['F1-score']:.4f}")

    output_file = ROOT_DIR / "runs" / "visual_analysis" / "visual_evaluation.csv"

    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print(f"Evaluation saved to: {output_file}")


if __name__ == "__main__":
    main()
