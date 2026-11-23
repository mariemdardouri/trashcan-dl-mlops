from pathlib import Path
import argparse
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def train(config_path, dataset_yaml=None):
    cfg = load_config(config_path)
    data_path = Path(dataset_yaml) if dataset_yaml else resolve_path(cfg["dataset_yaml"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML introuvable : {data_path}\n"
            "Indique le chemin avec --dataset-yaml."
        )

    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    out_cfg = cfg["output"]
    project = resolve_path(out_cfg["project"])

    print("=" * 60)
    print("EXPERIMENT 4 - YOLO11s")
    print("=" * 60)
    print("Dataset YAML :", data_path)
    print("Model :", cfg["model"])
    print("Epochs :", train_cfg["epochs"])
    print("Image size :", train_cfg["imgsz"])
    print("Batch :", train_cfg["batch"])

    model = YOLO(cfg["model"])
    results = model.train(
        data=str(data_path),
        **train_cfg,
        **aug_cfg,
        project=str(project),
        name=out_cfg["name"],
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    print("\nTRAINING FINISHED")
    print("Results :", save_dir)
    print("Best weights :", save_dir / "weights" / "best.pt")
    print("Last weights :", save_dir / "weights" / "last.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp4_underwater_augmentation.yaml"),
    )
    parser.add_argument("--dataset-yaml", default=None)
    args = parser.parse_args()
    train(args.config, args.dataset_yaml)
