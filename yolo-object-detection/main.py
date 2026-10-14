from pathlib import Path
import random
import shutil
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO



DATASET_DIR = Path("dataset")
WORK_DIR = Path("yolo_dataset")
RUNS_DIR = Path("runs")

MODEL_NAME = "yolo26n.pt"

EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 8
RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

random.seed(RANDOM_STATE)



IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root):
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_label(image_path):
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt",
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent / "labels" / f"{image_path.stem}.txt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(DATASET_DIR.rglob(f"{image_path.stem}.txt"))
    return matches[0] if matches else None


def find_existing_yaml():
    yaml_files = list(DATASET_DIR.rglob("*.yaml")) + list(DATASET_DIR.rglob("*.yml"))

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

            if isinstance(data, dict) and "names" in data:
                return yaml_file, data
        except Exception:
            pass

    return None, None



# ПОДГОТОВКА ДАТАСЕТА

def prepare_dataset():
    yaml_file, yaml_data = find_existing_yaml()

    if yaml_file:
        print(f"Найден data.yaml: {yaml_file}")

    images = find_images(DATASET_DIR)

    if not images:
        raise FileNotFoundError(
            "Не найдены изображения.\n"
            "Положите скачанный и распакованный датасет в папку 'dataset'."
        )

    pairs = []

    for image in images:
        label = find_label(image)

        if label is not None:
            pairs.append((image, label))

    if not pairs:
        raise FileNotFoundError(
            "Не найдена YOLO-разметка (*.txt).\n"
            "Нужен датасет с bounding-box разметкой в формате YOLO."
        )

    print(f"Найдено изображений с разметкой: {len(pairs)}")

    random.shuffle(pairs)

    train_end = int(len(pairs) * TRAIN_RATIO)
    val_end = train_end + int(len(pairs) * VAL_RATIO)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    for split, split_pairs in {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }.items():

        image_dir = WORK_DIR / "images" / split
        label_dir = WORK_DIR / "labels" / split

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image, label in split_pairs:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / f"{image.stem}.txt")

    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"Test:  {len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs, yaml_data



# КЛАССЫ

def get_class_names(yaml_data):
    if yaml_data and "names" in yaml_data:
        names = yaml_data["names"]

        if isinstance(names, dict):
            return [names[key] for key in sorted(names, key=lambda x: int(x))]

        if isinstance(names, list):
            return names

    class_ids = set()

    for label_file in (WORK_DIR / "labels").rglob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                if parts:
                    class_ids.add(int(parts[0]))

    max_class = max(class_ids)

    return [f"class_{i}" for i in range(max_class + 1)]


def create_data_yaml(class_names):
    data = {
        "path": str(WORK_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = WORK_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False
        )

    return yaml_path


def show_samples(class_names, max_images=12):
    image_files = list((WORK_DIR / "images" / "train").glob("*"))

    samples = []

    for image_path in image_files:
        label_path = WORK_DIR / "labels" / "train" / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        if lines:
            class_id = int(lines[0].split()[0])
            samples.append((image_path, class_id))

    samples = samples[:max_images]

    if not samples:
        return

    columns = 3
    rows = (len(samples) + columns - 1) // columns

    plt.figure(figsize=(15, 5 * rows))

    for i, (image_path, class_id) in enumerate(samples):
        image = Image.open(image_path)

        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
        plt.title(class_names[class_id])
        plt.axis("off")

    plt.tight_layout()

    output_path = WORK_DIR / "train_samples.png"
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Примеры изображений сохранены: {output_path}")


def train_model(data_yaml):
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(RUNS_DIR),
        name="lab3_yolo",
        exist_ok=True,
        seed=RANDOM_STATE,
        plots=True
    )

    return model


def evaluate_model(model, data_yaml):
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        plots=True
    )

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)

    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    return metrics


def predict_test_images(model):
    test_dir = WORK_DIR / "images" / "test"

    model.predict(
        source=str(test_dir),
        imgsz=IMAGE_SIZE,
        conf=0.25,
        save=True,
        project=str(RUNS_DIR),
        name="test_predictions",
        exist_ok=True
    )


def main():
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №3")
    print("Детекция объектов на изображении с использованием YOLO")
    print("=" * 60)

    train_pairs, val_pairs, test_pairs, yaml_data = prepare_dataset()

    class_names = get_class_names(yaml_data)

    print("\nКлассы:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    data_yaml = create_data_yaml(class_names)

    print(f"\nСоздан файл: {data_yaml}")

    show_samples(class_names)

    model = train_model(data_yaml)

    evaluate_model(model, data_yaml)

    predict_test_images(model)

    print("\n" + "=" * 60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Результаты обучения: {RUNS_DIR / 'lab3_yolo'}")
    print(f"Предсказания: {RUNS_DIR / 'test_predictions'}")


if __name__ == "__main__":
    main()
