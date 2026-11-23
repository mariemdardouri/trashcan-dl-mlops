from pathlib import Path
import argparse
import torch
import yaml
from rfdetr import RFDETRSmall


ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def train(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dataset_dir = resolve(cfg["dataset"]["work_dir"])
    output_dir = resolve(cfg["output"]["directory"])
    data_yaml = dataset_dir / "data.yaml"

    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA non disponible.")
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} absent. Lance d'abord src/prepare_exp5_dataset.py."
        )

    t = cfg["training"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 5 - RF-DETR SMALL")
    print("=" * 60)
    print("GPU :", torch.cuda.get_device_name(0))
    print("Dataset :", dataset_dir)
    print("Output :", output_dir)

    model = RFDETRSmall()
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        grad_accum_steps=t["grad_accum_steps"],
        lr=t["lr"],
        num_workers=t["num_workers"],
        amp_dtype=t["amp_dtype"],
        pin_memory=t["pin_memory"],
        persistent_workers=t["persistent_workers"],
        prefetch_factor=t["prefetch_factor"],
        scale_jitter=t["scale_jitter"],
        augmentation_backend=t["augmentation_backend"],
        eval_interval=t["eval_interval"],
        early_stopping=t["early_stopping"],
        early_stopping_patience=t["early_stopping_patience"],
        early_stopping_min_delta=t["early_stopping_min_delta"],
        checkpoint_interval=t["checkpoint_interval"],
        output_dir=str(output_dir),
        tensorboard=t["tensorboard"],
        wandb=t["wandb"],
        mlflow=t["mlflow"],
        clearml=t["clearml"],
        log_per_class_metrics=t["log_per_class_metrics"],
        seed=t["seed"],
    )
    print("\nEXPERIMENT 5 FINISHED")
    print("Inspecte les checkpoints et métriques dans :", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp5_rfdetr.yaml"),
    )
    args = parser.parse_args()
    train(args.config)
