
from pathlib import Path
from collections import Counter, defaultdict
import random
import shutil
import yaml

# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = ROOT / "data" / "processed" / "trashcan_yolo"
OUTPUT_ROOT = ROOT / "data" / "experiments" / "exp2_oversampling"

SOURCE_IMAGES = SOURCE_ROOT / "images" / "train"
SOURCE_LABELS = SOURCE_ROOT / "labels" / "train"

VAL_IMAGES = SOURCE_ROOT / "images" / "val"

SEED = 42
TARGET_ANNOTATIONS = 400
MAX_COPIES_PER_IMAGE = 10

CLASS_NAMES = [
    "trash_etc",
    "trash_plastic",
    "trash_metal",
    "trash_wood",
    "trash_fabric",
    "trash_fishing_gear",
    "trash_paper",
    "trash_rubber"
]

RARE_CLASSES = [
    "trash_fabric",
    "trash_paper",
    "trash_fishing_gear",
    "trash_rubber"
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

RARE_IDS = {
    CLASS_TO_ID[name]
    for name in RARE_CLASSES
}

random.seed(SEED)


# =====================================================
# 2. VERIFY INPUTS
# =====================================================

if not SOURCE_IMAGES.exists():
    raise FileNotFoundError(SOURCE_IMAGES)

if not SOURCE_LABELS.exists():
    raise FileNotFoundError(SOURCE_LABELS)

if not VAL_IMAGES.exists():
    raise FileNotFoundError(VAL_IMAGES)

if OUTPUT_ROOT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT_ROOT}. "
        "Inspect it before removing it."
    )

image_files = sorted([
    file for file in SOURCE_IMAGES.iterdir()
    if file.is_file()
])

if not image_files:
    raise ValueError("No training images found.")

print("Original training images:", len(image_files))


# =====================================================
# 3. READ YOLO LABELS
# =====================================================

image_labels = {}
class_counts = Counter()
candidates = defaultdict(list)

for image_path in image_files:

    label_path = SOURCE_LABELS / (image_path.stem + ".txt")

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    boxes = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:

            values = line.strip().split()

            if not values:
                continue

            class_id = int(values[0])

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(
                    f"Invalid class ID {class_id} in {label_path}"
                )

            boxes.append(class_id)
            class_counts[class_id] += 1

    image_labels[image_path.name] = boxes

    for class_id in set(boxes):
        if class_id in RARE_IDS:
            candidates[class_id].append(image_path.name)


# =====================================================
# 4. DISPLAY ORIGINAL DISTRIBUTION
# =====================================================

print("\nOriginal class distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name}: {class_counts[class_id]}")

print("\nNegative images:", sum(
    1 for boxes in image_labels.values() if len(boxes) == 0
))


# =====================================================
# 5. BUILD OVERSAMPLING PLAN
# =====================================================

original_counts = class_counts.copy()
new_counts = class_counts.copy()

copy_counts = Counter()
oversampling_plan = []

while True:

    classes_below_target = [
        class_id for class_id in RARE_IDS
        if new_counts[class_id] < TARGET_ANNOTATIONS
    ]

    if not classes_below_target:
        break

    # Select the rare class furthest below its target.
    selected_class = min(
        classes_below_target,
        key=lambda class_id:
            new_counts[class_id] / TARGET_ANNOTATIONS
    )

    available = [
        name for name in candidates[selected_class]
        if copy_counts[name] < MAX_COPIES_PER_IMAGE
    ]

    if not available:
        raise RuntimeError(
            f"Cannot reach target for "
            f"{CLASS_NAMES[selected_class]}. "
            "Increase MAX_COPIES_PER_IMAGE or reduce target."
        )

    # Prefer images that have been copied fewer times.
    minimum_copies = min(copy_counts[name] for name in available)

    balanced_candidates = [
        name for name in available
        if copy_counts[name] == minimum_copies
    ]

    selected_image = random.choice(balanced_candidates)

    oversampling_plan.append(selected_image)
    copy_counts[selected_image] += 1

    # Update counts for every class in this image.
    for class_id in image_labels[selected_image]:
        new_counts[class_id] += 1


# =====================================================
# 6. CREATE OUTPUT DIRECTORIES
# =====================================================

OUTPUT_IMAGES = OUTPUT_ROOT / "images" / "train"
OUTPUT_LABELS = OUTPUT_ROOT / "labels" / "train"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)


# =====================================================
# 7. COPY ALL ORIGINAL TRAINING FILES
# =====================================================

for image_path in image_files:

    label_path = SOURCE_LABELS / (image_path.stem + ".txt")

    shutil.copy2(image_path, OUTPUT_IMAGES / image_path.name)
    shutil.copy2(label_path, OUTPUT_LABELS / label_path.name)


# =====================================================
# 8. CREATE REAL DUPLICATE SAMPLES
# =====================================================

repeat_numbers = Counter()

for image_name in oversampling_plan:

    source_image = SOURCE_IMAGES / image_name
    source_label = SOURCE_LABELS / (source_image.stem + ".txt")

    repeat_numbers[image_name] += 1
    repeat_number = repeat_numbers[image_name]

    new_stem = f"{source_image.stem}__os{repeat_number:03d}"

    new_image = OUTPUT_IMAGES / (
        new_stem + source_image.suffix
    )

    new_label = OUTPUT_LABELS / (new_stem + ".txt")

    shutil.copy2(source_image, new_image)
    shutil.copy2(source_label, new_label)


# =====================================================
# 9. VERIFY OUTPUT
# =====================================================

output_images = sorted([
    file for file in OUTPUT_IMAGES.iterdir()
    if file.is_file()
])

output_labels = sorted([
    file for file in OUTPUT_LABELS.iterdir()
    if file.is_file()
])

if len(output_images) != len(output_labels):
    raise RuntimeError("Image/label file count mismatch.")

final_counts = Counter()

for label_path in output_labels:

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:

            values = line.strip().split()

            if values:
                final_counts[int(values[0])] += 1

print("\nFinal class distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name}: "
        f"{original_counts[class_id]} -> "
        f"{final_counts[class_id]}"
    )

print("\nOriginal images:", len(image_files))
print("Added duplicate images:", len(oversampling_plan))
print("Final training images:", len(output_images))
print("Final training labels:", len(output_labels))


# =====================================================
# 10. CREATE YOLO DATA YAML
# =====================================================

data_yaml = {
    "path": OUTPUT_ROOT.as_posix(),
    "train": "images/train",
    "val": VAL_IMAGES.resolve().as_posix(),
    "names": {
        index: name
        for index, name in enumerate(CLASS_NAMES)
    }
}

with open(
    OUTPUT_ROOT / "data.yaml",
    "w",
    encoding="utf-8"
) as file:
    yaml.safe_dump(
        data_yaml,
        file,
        sort_keys=False,
        allow_unicode=True
    )


# =====================================================
# 11. SAVE EXPERIMENT REPORT
# =====================================================

report_path = OUTPUT_ROOT / "oversampling_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("EXPERIMENT 2 - OVERSAMPLING REPORT\n\n")
    file.write(f"Original images: {len(image_files)}\n")
    file.write(f"Added copies: {len(oversampling_plan)}\n")
    file.write(f"Final images: {len(output_images)}\n\n")

    for class_id, class_name in enumerate(CLASS_NAMES):
        file.write(
            f"{class_name}: "
            f"{original_counts[class_id]} -> "
            f"{final_counts[class_id]}\n"
        )

print("\nExperiment 2 dataset preparation completed.")
print("Dataset:", OUTPUT_ROOT)
print("YOLO config:", OUTPUT_ROOT / "data.yaml")
print("Report:", report_path)
print("\nOriginal baseline dataset was not modified.")
