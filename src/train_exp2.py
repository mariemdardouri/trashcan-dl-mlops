
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "exp2_oversampling.yaml"

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


# ==========================================
# 2. MAIN TRAINING FUNCTION
# ==========================================

def main():

    print("=" * 55)
    print("TRASHCAN - EXPERIMENT 2: CLASS IMBALANCE")
    print("=" * 55)

    # Load experiment configuration
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    experiment_name = config["experiment_name"]

    model_name = config["model"]["pretrained"]

    training = config["training"]

    dataset_root = (
        PROJECT_ROOT
        / config["dataset"]["experiment"]
    )

    data_yaml = dataset_root / "data.yaml"

    # Verify dataset
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    print("Experiment:", experiment_name)
    print("Dataset:", data_yaml)

    # Verify GPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check your GPU environment."
        )

    device = training["device"]

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    # Load pretrained model
    print("\nLoading pretrained YOLO11n model...")

    model = YOLO(model_name)

    print("Model loaded successfully!")

    # Unique experiment name
    run_name = "trashcan_yolo11n_exp2_oversampling"

    print("\nStarting Experiment 2 training...\n")

    results = model.train(

        data=str(data_yaml),

        epochs=training["epochs"],
        imgsz=training["imgsz"],
        batch=training["batch"],

        device=device,
        workers=training["workers"],

        patience=training["patience"],
        seed=training["seed"],

        project=str(RUNS_DIR),
        name=run_name,

        pretrained=True,
        plots=True,
        save=True,
        verbose=True,

        exist_ok=False
    )

    print("\nExperiment 2 training completed!")

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
