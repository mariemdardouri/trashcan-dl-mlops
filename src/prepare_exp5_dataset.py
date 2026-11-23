from pathlib import Path
import argparse
import shutil
import yaml


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "trash_etc", "trash_fabric", "trash_fishing_gear", "trash_metal",
    "trash_paper", "trash_plastic", "trash_rubber", "trash_wood",
]


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def prepare(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source = resolve(cfg["dataset"]["source_dir"])
    target = resolve(cfg["dataset"]["work_dir"])

    expected = [
        source / "images" / "train", source / "labels" / "train",
        source / "images" / "val", source / "labels" / "val",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError("Éléments dataset manquants :\n" + "\n".join(missing))

    target.mkdir(parents=True, exist_ok=True)
    mapping = {"train": "train", "valid": "val"}
    for target_split, source_split in mapping.items():
        for folder in ("images", "labels"):
            src = source / folder / source_split
            dst = target / target_split / folder
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
            print(f"Copié : {src} -> {dst}")

    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "nc": 8,
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    yaml_path = target / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("\nDataset RF-DETR prêt :", target)
    print("YAML :", yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp5_rfdetr.yaml"),
    )
    args = parser.parse_args()
    prepare(args.config)
