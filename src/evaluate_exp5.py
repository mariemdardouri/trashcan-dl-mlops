from pathlib import Path
import argparse
import json
import pandas as pd
import yaml
from rfdetr import RFDETRSmall


ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def evaluate(config_path, checkpoint, split="val"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dataset_dir = resolve(cfg["dataset"]["work_dir"])
    output_dir = resolve(cfg["output"]["directory"])
    checkpoint_path = Path(checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    if not (dataset_dir / "data.yaml").exists():
        raise FileNotFoundError(
            f"{dataset_dir / 'data.yaml'} absent. Prépare d'abord le dataset."
        )

    # API evaluate() et from_checkpoint() vérifiées dans la cellule 20 du notebook.
    model = RFDETRSmall.from_checkpoint(str(checkpoint_path))
    metrics = model.evaluate(dataset_dir=str(dataset_dir), split=split)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"evaluation_{split}.json"
    csv_path = output_dir / f"evaluation_{split}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    if isinstance(metrics, dict):
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    else:
        pd.DataFrame([{"metrics": str(metrics)}]).to_csv(csv_path, index=False)

    print("Évaluation terminée.")
    print("Métriques :", metrics)
    print("JSON :", json_path)
    print("CSV :", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp5_rfdetr.yaml"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.split)
