
from pathlib import Path
from ultralytics import YOLO


def main():

    # 1. Project paths
    ROOT_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = (
        ROOT_DIR / "runs" / "detect" /
        "trashcan_yolo11n_baseline" / "weights" / "best.pt"
    )

    DATA_YAML = (
        ROOT_DIR / "data" / "processed" /
        "trashcan_yolo" / "data.yaml"
    )

    # 2. Load the best model
    print("Loading best model...")
    model = YOLO(str(MODEL_PATH))

    # 3. Evaluate the model
    print("\nStarting validation...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT_DIR / "runs" / "evaluation"),
        name="baseline_evaluation",
        exist_ok=True
    )

    # 4. Global metrics
    print("\n========== GLOBAL METRICS ==========")

    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")

    # 5. Per-class metrics
    print("\n========== CLASS METRICS ==========")

    for class_id, class_name in model.names.items():

        print(f"\nClass: {class_name}")
        print(f"mAP50:     {metrics.box.ap50[class_id]:.4f}")
        print(f"mAP50-95:  {metrics.box.ap[class_id]:.4f}")



    
    # 6. Confusion matrix diagnostic
    import csv

    confusion_matrix = metrics.confusion_matrix.matrix

    class_names = [
        model.names[i] for i in range(len(model.names))
    ] + ["background"]

    print("\n========== CONFUSION MATRIX DEBUG ==========")

    print("Matrix shape:", confusion_matrix.shape)
    print("Total matrix values:", confusion_matrix.sum())

    print("\nRaw confusion matrix:")
    print(confusion_matrix)

    print("\nRow sums:")
    for i, row in enumerate(confusion_matrix):
        print(class_names[i], ":", row.sum())

    print("\nColumn sums:")
    for i in range(len(class_names)):
        print(class_names[i], ":", confusion_matrix[:, i].sum())

    # Export CSV
    csv_path = (
        ROOT_DIR / "runs" / "evaluation" /
        "baseline_evaluation" / "confusion_matrix.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Predicted / Actual"] + class_names)

        for i, row in enumerate(confusion_matrix):
            writer.writerow([class_names[i]] + list(row))

    print("\nConfusion matrix exported:", csv_path)



    print("\nEvaluation completed!")
    print("Results saved in:")
    print(ROOT_DIR / "runs" / "evaluation" / "baseline_evaluation")


if __name__ == "__main__":
    main()
