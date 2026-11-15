
from pathlib import Path
import torch
from ultralytics import YOLO


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trashcan_yolo"
    / "data.yaml"
)

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


# ==========================================
# 2. TRAINING CONFIGURATION
# ==========================================

MODEL_NAME = "yolo11n.pt"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 15
SEED = 42
WORKERS = 4


# ==========================================
# 3. MAIN TRAINING FUNCTION
# ==========================================

def main():

    print("=" * 50)
    print("TRASHCAN - YOLO OBJECT DETECTION")
    print("=" * 50)

    # Check dataset configuration
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    print("Dataset:", DATA_YAML)

    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check your GPU environment."
        )

    device = 0

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    # Load pretrained model
    print("\nLoading pretrained YOLO11n model...")

    model = YOLO(MODEL_NAME)

    print("Model loaded successfully!")

    # Start training
    print("\nStarting training...\n")

    results = model.train(
        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=device,
        workers=WORKERS,

        patience=PATIENCE,
        seed=SEED,

        project=str(RUNS_DIR),
        name="trashcan_yolo11n_baseline",

        pretrained=True,
        plots=True,
        save=True,
        verbose=True
    )

    print("\nTraining completed!")

    print("Results directory:")
    print(results.save_dir)

    print("\nBest model:")
    print(Path(results.save_dir) / "weights" / "best.pt")


# ==========================================
# 4. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
