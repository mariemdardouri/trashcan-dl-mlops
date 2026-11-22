
from pathlib import Path
import random
import shutil
import csv

ROOT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT_DIR / "data/processed/trashcan_yolo/images/val"
LABEL_DIR = ROOT_DIR / "data/processed/trashcan_yolo/labels/val"

OUTPUT_DIR = ROOT_DIR / "runs/visual_analysis"
SAMPLE_DIR = OUTPUT_DIR / "sample_images"

CLASS_NAMES = {
    0: "trash_etc",
    1: "trash_plastic",
    2: "trash_metal",
    3: "trash_wood",
    4: "trash_fabric",
    5: "trash_fishing_gear",
    6: "trash_paper",
    7: "trash_rubber"
}

SAMPLES_PER_CLASS = 2
NEGATIVE_SAMPLES = 2
SEED = 42


def main():

    random.seed(SEED)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    images = list(IMAGE_DIR.glob("*.jpg"))

    selected = {}

    # Select images containing each class
    for class_id, class_name in CLASS_NAMES.items():

        candidates = []

        for image_path in images:

            label_path = LABEL_DIR / (image_path.stem + ".txt")

            if not label_path.exists():
                continue

            with open(label_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) >= 5 and int(float(parts[0])) == class_id:
                    candidates.append(image_path)
                    break

        random.shuffle(candidates)

        selected[class_id] = candidates[:SAMPLES_PER_CLASS]

    # Select negative images (empty labels)
    negative_candidates = []

    for image_path in images:

        label_path = LABEL_DIR / (image_path.stem + ".txt")

        if label_path.exists() and label_path.stat().st_size == 0:
            negative_candidates.append(image_path)

    random.shuffle(negative_candidates)

    # Avoid selecting the same image more than once
    selected_images = []

    for class_images in selected.values():
        for image_path in class_images:
            if image_path not in selected_images:
                selected_images.append(image_path)

    for image_path in negative_candidates:

        if len([
            path for path in selected_images
            if (LABEL_DIR / (path.stem + ".txt")).stat().st_size == 0
        ]) >= NEGATIVE_SAMPLES:
            break

        if image_path not in selected_images:
            selected_images.append(image_path)

    # Copy images and create manifest
    manifest_path = OUTPUT_DIR / "sample_manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "image",
            "ground_truth_classes",
            "number_of_objects"
        ])

        for image_path in selected_images:

            label_path = LABEL_DIR / (image_path.stem + ".txt")

            shutil.copy2(image_path, SAMPLE_DIR / image_path.name)

            class_ids = []

            if label_path.exists():

                with open(label_path, "r", encoding="utf-8") as label_file:

                    for line in label_file:

                        parts = line.strip().split()

                        if len(parts) >= 5:
                            class_ids.append(int(float(parts[0])))

            ground_truth = [
                CLASS_NAMES[class_id]
                for class_id in sorted(set(class_ids))
            ]

            writer.writerow([
                image_path.name,
                "; ".join(ground_truth) if ground_truth else "negative",
                len(class_ids)
            ])

    print("\nVisual sample prepared!")
    print("Number of images:", len(selected_images))
    print("Sample folder:", SAMPLE_DIR)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
