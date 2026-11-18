
from pathlib import Path
import csv
from ultralytics import YOLO


def main():

    # 1. Project paths
    ROOT_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = (
        ROOT_DIR / "runs" / "detect" /
        "trashcan_yolo11n_exp2_oversampling" / "weights" / "best.pt"
    )

    # Use the original validation dataset for a fair comparison
    DATA_YAML = (
        ROOT_DIR / "data" / "processed" /
        "trashcan_yolo" / "data.yaml"
    )

    OUTPUT_DIR = ROOT_DIR / "runs" / "evaluation"
    OUTPUT_NAME = "exp2_oversampling_evaluation"

    # 2. Check paths
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    # 3. Load the best model
    print("Loading Experiment 2 best model...")
    model = YOLO(str(MODEL_PATH))

    # 4. Evaluate
    print("\nStarting Experiment 2 validation...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR),
        name=OUTPUT_NAME,
        exist_ok=True
    )

    # 5. Global metrics
    print("\n========== EXPERIMENT 2 GLOBAL METRICS ==========")

    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")

    # 6. Per-class metrics
    print("\n========== EXPERIMENT 2 CLASS METRICS ==========")

    for class_id, class_name in model.names.items():
        print(f"\nClass: {class_name}")
        print(f"mAP50:     {metrics.box.ap50[class_id]:.4f}")
        print(f"mAP50-95:  {metrics.box.ap[class_id]:.4f}")

    # 7. Confusion matrix
    confusion_matrix = metrics.confusion_matrix.matrix

    class_names = [
        model.names[i] for i in range(len(model.names))
    ] + ["background"]

    print("\n========== CONFUSION MATRIX DEBUG ==========")
    print("Matrix shape:", confusion_matrix.shape)
    print("Total matrix values:", confusion_matrix.sum())
    print("\nRaw confusion matrix:")
    print(confusion_matrix)

    # 8. Export confusion matrix
    csv_path = OUTPUT_DIR / OUTPUT_NAME / "confusion_matrix.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Predicted / Actual"] + class_names)

        for i, row in enumerate(confusion_matrix):
            writer.writerow([class_names[i]] + list(row))

    print("\nConfusion matrix exported:", csv_path)

    print("\nExperiment 2 evaluation completed!")
    print("Results saved in:")
    print(OUTPUT_DIR / OUTPUT_NAME)


if __name__ == "__main__":
    main()
