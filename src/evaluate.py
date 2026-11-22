
from pathlib import Path
import argparse
import csv

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_YAML = (
    ROOT_DIR / "data" / "processed" /
    "trashcan_yolo" / "data.yaml"
)

MODEL_PATHS = {
    "baseline": (
        ROOT_DIR / "runs" / "detect" /
        "trashcan_yolo11n_baseline" / "weights" / "best.pt"
    ),
    "exp2": (
        ROOT_DIR / "runs" / "detect" /
        "trashcan_yolo11n_exp2_oversampling" / "weights" / "best.pt"
    ),
    "exp3": (
        ROOT_DIR / "runs" / "detect" /
        "trashcan_yolo11n_exp3_augmentation-2" / "weights" / "best.pt"
    ),
    "exp4": (
        ROOT_DIR / "runs" / "detect" /
        "trashcan_yolo11s_exp4" / "weights" / "best.pt"
    )
}


def evaluate_model(model_name):

    model_path = MODEL_PATHS[model_name]

    output_dir = (
        ROOT_DIR / "runs" / "evaluation" /
        f"{model_name}_evaluation"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nEvaluating model: {model_name}")
    print("Loading:", model_path)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT_DIR / "runs" / "evaluation"),
        name=f"{model_name}_evaluation",
        exist_ok=True
    )

    # Global metrics
    precision = metrics.box.mp
    recall = metrics.box.mr

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    print("\n========== GLOBAL METRICS ==========")
    print(f"Model:      {model_name}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")

    # Per-class metrics
    class_metrics_path = output_dir / "class_metrics.csv"

    with open(class_metrics_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Class", "Precision", "Recall",
            "F1-score", "mAP50", "mAP50-95"
        ])

        print("\n========== PER-CLASS METRICS ==========")

        for class_id, class_name in model.names.items():

            class_precision = metrics.box.p[class_id]
            class_recall = metrics.box.r[class_id]
            class_f1 = metrics.box.f1[class_id]

            map50 = metrics.box.ap50[class_id]
            map5095 = metrics.box.ap[class_id]

            print(f"\n{class_name}")
            print(f"Precision: {class_precision:.4f}")
            print(f"Recall:    {class_recall:.4f}")
            print(f"F1-score:  {class_f1:.4f}")
            print(f"mAP50:     {map50:.4f}")
            print(f"mAP50-95:  {map5095:.4f}")

            writer.writerow([
                class_name,
                class_precision,
                class_recall,
                class_f1,
                map50,
                map5095
            ])

    # Inference speed
    speed = metrics.speed

    inference_ms = speed["inference"]
    total_ms = sum(speed.values())

    fps = 1000 / inference_ms if inference_ms > 0 else 0
    total_fps = 1000 / total_ms if total_ms > 0 else 0

    print("\n========== INFERENCE SPEED ==========")
    print(f"Inference time: {inference_ms:.2f} ms/image")
    print(f"Inference FPS:  {fps:.2f}")
    print(f"Total FPS:      {total_fps:.2f}")

    # Confusion matrix
    matrix = metrics.confusion_matrix.matrix

    class_names = [
        model.names[i] for i in range(len(model.names))
    ] + ["background"]

    confusion_path = output_dir / "confusion_matrix.csv"

    with open(confusion_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted"] + class_names)

        for i, row in enumerate(matrix):
            writer.writerow([class_names[i]] + list(row))

    print("\nClass metrics:", class_metrics_path)
    print("Confusion matrix:", confusion_path)
    print("Evaluation completed!")


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate TrashCan YOLO models"
    )

    parser.add_argument(
        "--model",
        choices=["baseline", "exp2", "exp3", "exp4", "all"],
        required=True,
        help="Choose the model to evaluate"
    )

    args = parser.parse_args()

    if args.model == "all":
        for model_name in ["baseline", "exp2", "exp3", "exp4"]:
            evaluate_model(model_name)
    else:
        evaluate_model(args.model)


if __name__ == "__main__":
    main()
