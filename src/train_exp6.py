from pathlib import Path
import argparse
import sys
import yaml
import torch


ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def train(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    repo_dir = resolve(cfg["architecture"]["repository_dir"])
    model_yaml = resolve(cfg["architecture"]["model_yaml"])
    dataset_dir = resolve(cfg["dataset"]["work_dir"])
    data_yaml = dataset_dir / "data.yaml"

    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA non disponible.")
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Dépôt YOLO-Marine absent : {repo_dir}\n"
            "Clone-le selon le README de cette expérience."
        )
    if not model_yaml.exists():
        raise FileNotFoundError(f"Architecture YAML absente : {model_yaml}")
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} absent. Lance d'abord src/augment_exp6.py."
        )

    # Charger l'implémentation Ultralytics modifiée du dépôt YOLO-Marine.
    repo_path = str(repo_dir)
    if repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)

    from ultralytics import YOLO

    train_cfg = cfg["training"]
    out_cfg = cfg["output"]
    output_project = resolve(out_cfg["project"])

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data["path"] = str(dataset_dir)
    data["train"] = "images/train"
    data["val"] = "images/val"
    data_yaml.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("GPU :", torch.cuda.get_device_name(0))
    print("Architecture : YOLO-Marine MixStructure")
    print("Dataset :", dataset_dir)
    print("Classes :", len(data["names"]))
    print("Model YAML :", model_yaml)

    model = YOLO(str(model_yaml))
    model.load(cfg["architecture"]["pretrained_weights"])

    results = model.train(
        data=str(data_yaml),
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        workers=train_cfg["workers"],
        device=train_cfg["device"],
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        weight_decay=train_cfg["weight_decay"],
        cos_lr=train_cfg["cos_lr"],
        patience=train_cfg["patience"],
        amp=train_cfg["amp"],
        cache=train_cfg["cache"],
        seed=cfg["experiment"]["seed"],
        deterministic=train_cfg["deterministic"],
        hsv_h=train_cfg["hsv_h"],
        hsv_s=train_cfg["hsv_s"],
        hsv_v=train_cfg["hsv_v"],
        degrees=train_cfg["degrees"],
        translate=train_cfg["translate"],
        scale=train_cfg["scale"],
        fliplr=train_cfg["fliplr"],
        flipud=train_cfg["flipud"],
        mosaic=train_cfg["mosaic"],
        mixup=train_cfg["mixup"],
        close_mosaic=train_cfg["close_mosaic"],
        project=str(output_project),
        name=out_cfg["name"],
        exist_ok=True,
        save=True,
        save_period=train_cfg["save_period"],
        plots=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    print("Entraînement terminé.")
    print("Résultats :", save_dir)
    print("Best :", save_dir / "weights" / "best.pt")
    print("Last :", save_dir / "weights" / "last.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp6_yolo_marine.yaml"),
    )
    args = parser.parse_args()
    train(args.config)
