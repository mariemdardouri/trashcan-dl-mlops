
from pathlib import Path
import argparse
from ultralytics import YOLO


def main():

    # 1. Project paths
    ROOT_DIR = Path(__file__).resolve().parent.parent

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
        )
    }

    # 2. Arguments
    parser = argparse.ArgumentParser(
        description="Run TrashCan YOLO predictions"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image or folder of images"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "exp2", "exp3", "all"],
        default="all",
        help="Model to use for prediction"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device (0) or CPU"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/predictions",
        help="Output directory for prediction results"
    )

    args = parser.parse_args()

    # 3. Check source
    source_path = Path(args.source)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Image or folder not found: {source_path}"
        )

    # 4. Select models
    if args.model == "all":
        selected_models = list(MODEL_PATHS.keys())
    else:
        selected_models = [args.model]

    # 5. Output directory
    OUTPUT_DIR = ROOT_DIR / args.output

    print("\n====================================")
    print("TRASHCAN OBJECT DETECTION")
    print("====================================")
    print(f"Source: {source_path}")
    print(f"Models: {selected_models}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Image size: {args.imgsz}")

    # 6. Run predictions
    for model_name in selected_models:

        model_path = MODEL_PATHS[model_name]

        if not model_path.exists():
            print(f"\nWARNING: Model not found: {model_path}")
            continue

        print(f"\n========== {model_name.upper()} ==========")

        model = YOLO(str(model_path))

        results = model.predict(
            source=str(source_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(OUTPUT_DIR),
            name=model_name,
            exist_ok=True
        )

        total_detections = 0

        for result in results:
            if result.boxes is not None:
                total_detections += len(result.boxes)

        print(f"Images processed: {len(results)}")
        print(f"Total detections: {total_detections}")
        print(f"Results saved in: {OUTPUT_DIR / model_name}")

    print("\nAll predictions completed!")


if __name__ == "__main__":
    main()
