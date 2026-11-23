from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "exp6_combined.yaml"

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


# ==========================================
# 2. MAIN TRAINING FUNCTION
# ==========================================

def main():

    print("=" * 60)
    print("TRASHCAN - EXPERIMENT 6: COMBINED TRAINING")
    print("=" * 60)

    # Load configuration
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    experiment_name = config["experiment_name"]
    model_name = config["model"]["pretrained"]
    training = config["training"]
    augmentation = config["augmentation"]

    # Resolve dataset path
    data_yaml = PROJECT_ROOT / config["dataset"]["path"]

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_yaml}"
        )

    # Verify the dataset configuration
    with open(data_yaml, "r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)

    train_path = (
        Path(dataset_config["path"]) / dataset_config["train"]
    )

    val_path = Path(dataset_config["val"])

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training images directory not found: {train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation images directory not found: {val_path}"
        )

    print("Experiment:", experiment_name)
    print("Model:", model_name)
    print("Dataset configuration:", data_yaml)
    print("Training images:", train_path)
    print("Validation images:", val_path)

    # Verify GPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check your GPU environment."
        )

    device = training["device"]

    print("\nGPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    # Load pretrained model
    print("\nLoading pretrained YOLO11n model...")

    model = YOLO(model_name)

    print("Model loaded successfully!")

    # Display augmentation parameters
    print("\nAugmentation parameters:")

    for name, value in augmentation.items():
        print(f"{name}: {value}")

    # Training
    run_name = "trashcan_yolo11n_exp6_combined"

    print("\nStarting Experiment 6 training...\n")

    results = model.train(

        # Dataset
        data=str(data_yaml),

        # Training parameters
        epochs=training["epochs"],
        imgsz=training["imgsz"],
        batch=training["batch"],
        device=device,
        workers=training["workers"],
        patience=training["patience"],
        seed=training["seed"],

        # Augmentation
        **augmentation,

        # Output
        project=str(RUNS_DIR),
        name=run_name,

        pretrained=True,
        plots=True,
        save=True,
        verbose=True,

        exist_ok=False
    )

    print("\nExperiment 6 training completed!")

    print("Results directory:")
    print(results.save_dir)

    best_model = (
        Path(results.save_dir)
        / "weights"
        / "best.pt"
    )

    print("\nBest model:")
    print(best_model)


# ==========================================
# 3. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()