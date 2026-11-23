from pathlib import Path
import argparse
import yaml
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def evaluate(config_path, model_path=None, dataset_yaml=None):
    cfg = load_config(config_path)
    out_cfg = cfg["output"]
    data_path = Path(dataset_yaml) if dataset_yaml else resolve_path(cfg["dataset_yaml"])

    weights = Path(model_path) if model_path else (
        resolve_path(out_cfg["project"]) / out_cfg["name"] / "weights" / "best.pt"
    )
    if not weights.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML introuvable : {data_path}")

    eval_root = resolve_path(out_cfg["evaluation_project"])
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_path),
        split="val",
        imgsz=cfg["training"]["imgsz"],
        batch=cfg["training"]["batch"],
        device=cfg["training"]["device"],
        workers=cfg["training"]["workers"],
        plots=True,
        project=str(eval_root),
        name=out_cfg["evaluation_name"],
        exist_ok=True,
    )

    rows = [{
        "class": "ALL",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }]
    for class_id, ap in enumerate(metrics.box.maps):
        rows.append({
            "class": model.names[class_id],
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50_95": float(ap),
        })

    output_dir = eval_root / out_cfg["evaluation_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("Evaluation terminée.")
    print("Poids :", weights)
    print("CSV :", csv_path)
    print("Precision :", float(metrics.box.mp))
    print("Recall :", float(metrics.box.mr))
    print("mAP50 :", float(metrics.box.map50))
    print("mAP50-95 :", float(metrics.box.map))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp4_underwater_augmentation.yaml"),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset-yaml", default=None)
    args = parser.parse_args()
    evaluate(args.config, args.model, args.dataset_yaml)
