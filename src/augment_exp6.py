from pathlib import Path
from collections import defaultdict
import argparse
import random
import re
import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def read_labels(path):
    annotations = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                annotations.append([
                    int(parts[0]), float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])
                ])
    return annotations


def get_video_frame(filename):
    stem = Path(filename).stem
    match = re.search(r"^(.*)_frame(\d+)", stem, re.IGNORECASE)
    return (match.group(1), int(match.group(2))) if match else (stem, 0)


def augment_image(image, annotations):
    img = image.copy()
    boxes = [ann.copy() for ann in annotations]

    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        for box in boxes:
            box[1] = 1.0 - box[1]

    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-18, 18)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-5, 5)) % 180
    hsv[:, :, 1] *= random.uniform(0.85, 1.15)
    hsv[:, :, 2] *= random.uniform(0.90, 1.10)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.15:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, boxes


def run(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source = resolve(cfg["dataset"]["source_dir"])
    work = resolve(cfg["dataset"]["work_dir"])

    # Copier une seule fois le dataset source pour ne jamais modifier la source DVC.
    if not work.exists():
        import shutil
        shutil.copytree(source, work)
    else:
        print("Dataset de travail déjà présent :", work)

    image_dir = work / "images" / "train"
    label_dir = work / "labels" / "train"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Structure YOLO attendue : images/train et labels/train.")

    prefix_files = list(image_dir.glob("exp6aug_*"))
    if prefix_files:
        raise RuntimeError(
            "Des images exp6aug_* existent déjà. Ne relance pas l'augmentation "
            "sur le même dossier : supprime le dossier de travail puis relance."
        )

    aug_cfg = cfg["augmentation"]
    random.seed(aug_cfg["seed"])
    np.random.seed(aug_cfg["seed"])
    rare_classes = {int(k): v for k, v in aug_cfg["rare_classes"].items()}

    candidates = defaultdict(list)
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        labels = read_labels(label_dir / f"{image_path.stem}.txt")
        present_classes = {ann[0] for ann in labels}
        video_id, frame_number = get_video_frame(image_path.name)
        for class_id in rare_classes:
            if class_id in present_classes:
                candidates[class_id].append((video_id, frame_number, image_path))

    representatives = set()
    min_gap = int(aug_cfg["min_frame_gap"])
    for class_id, samples in candidates.items():
        by_video = defaultdict(list)
        for video_id, frame_number, image_path in samples:
            by_video[video_id].append((frame_number, image_path))
        for frames in by_video.values():
            frames.sort(key=lambda item: item[0])
            last_frame = None
            for frame_number, image_path in frames:
                if last_frame is None or frame_number - last_frame >= min_gap:
                    representatives.add(image_path)
                    last_frame = frame_number

    generated = 0
    for image_path in sorted(representatives):
        label_path = label_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        annotations = read_labels(label_path)
        if image is None or not annotations:
            continue

        for variant in range(1, int(aug_cfg["variants_per_image"]) + 1):
            aug_image, aug_boxes = augment_image(image, annotations)
            new_stem = f"exp6aug_{image_path.stem}_v{variant}"
            if not cv2.imwrite(str(image_dir / f"{new_stem}.jpg"), aug_image):
                raise RuntimeError(f"Échec écriture image : {new_stem}")
            with open(label_dir / f"{new_stem}.txt", "w", encoding="utf-8") as f:
                for class_id, x, y, w, h in aug_boxes:
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            generated += 1

    print("Images représentatives :", len(representatives))
    print("Nouvelles images générées :", generated)
    print("Validation inchangée.")
    print("Dataset de travail :", work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp6_yolo_marine.yaml"),
    )
    args = parser.parse_args()
    run(args.config)
