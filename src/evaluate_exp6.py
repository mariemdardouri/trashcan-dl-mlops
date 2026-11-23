from pathlib import Path
import argparse
import sys
import yaml
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def evaluate(config_path, model_path=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    repo_dir = resolve(cfg["architecture"]["repository_dir"])
    repo_path = str(repo_dir)
    if repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)
    from ultralytics import YOLO

    data_dir = resolve(cfg["dataset"]["work_dir"])
    data_yaml = data_dir / "data.yaml"
    out = cfg["output"]
    weights = Path(model_path) if model_path else (
        resolve(out["project"]) / out["name"] / "weights" / "best.pt"
    )
    if not weights.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML introuvable : {data_yaml}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=cfg["training"]["imgsz"],
        batch=cfg["training"]["batch"],
        device=cfg["training"]["device"],
        workers=cfg["training"]["workers"],
        plots=True,
        project=str(resolve(out["evaluation_project"])),
        name=out["evaluation_name"],
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

    output_dir = resolve(out["evaluation_project"]) / out["evaluation_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Évaluation terminée :", csv_path)
    print("Precision :", float(metrics.box.mp))
    print("Recall :", float(metrics.box.mr))
    print("mAP50 :", float(metrics.box.map50))
    print("mAP50-95 :", float(metrics.box.map))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp6_yolo_marine.yaml"),
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    evaluate(args.config, args.model)
