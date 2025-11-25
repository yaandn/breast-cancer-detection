import os
import cv2
import yaml
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

SOURCE_DIR = "dataset_augmented"
OUTPUT_DIR = "datasets/mamografias"

CLASSES = {"benign": 0, "malignant": 1, "normal": 2}

for sub in [
    "images/train", "images/val",
    "labels/train", "labels/val",
    "masks/train", "masks/val"
]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

image_paths = []

for class_name in CLASSES.keys():
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.exists(class_path):
        print(f"Pasta não encontrada: {class_path}")
        continue

    img_files = sorted(
        glob(os.path.join(class_path, "*.png")) +
        glob(os.path.join(class_path, "*.jpg")) +
        glob(os.path.join(class_path, "*.jpeg"))
    )

    print(f"\nClasse: {class_name} ({len(img_files)} imagens)")

    for img_path in img_files:

        if "_mask" in img_path.lower():
            continue

        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]

        mask_candidates = [
            f for f in os.listdir(class_path)
            if base in f and "_mask" in f.lower()
        ]

        if not mask_candidates:
            print(f"Sem máscara correspondente para {img_name}")
            continue

        mask_path = os.path.join(class_path, mask_candidates[0])

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Erro ao ler: {img_name} ou máscara")
            continue

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        h, w = mask.shape

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_content = ""

        if class_name == "normal":
            if len(contours) == 0:
                label_content = ""
                print(f"NORMAL sem contornos: {img_name}")
            else:
                print(f"WARNING: NORMAL com contornos")
                label_content = ""

        else:
            if len(contours) == 0:
                print(f"{img_name}: Nenhum contorno encontrado")
                continue

            print(f"✔ {img_name}: {len(contours)} contorno(s) encontrado(s)")

            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)

                if bw < 10 or bh < 10:
                    continue

                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                width = bw / w
                height = bh / h

                label_content += f"{CLASSES[class_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

            if not label_content.strip():
                continue

        out_img = os.path.join(OUTPUT_DIR, "images/train", img_name)
        out_mask = os.path.join(OUTPUT_DIR, "masks/train", base + "_mask.png")
        out_label = os.path.join(OUTPUT_DIR, "labels/train", base + ".txt")

        shutil.copy(img_path, out_img)
        shutil.copy(mask_path, out_mask)

        with open(out_label, "w") as f:
            f.write(label_content)

        image_paths.append(out_img)

print(f"\nTotal de imagens processadas: {len(image_paths)}")

if len(image_paths) == 0:
    raise ValueError("Nenhuma imagem processada: máscaras não econtradas")

train_files, val_files = train_test_split(image_paths, test_size=0.2, random_state=42)

def move_to_val(files):
    for img_train_path in files:

        img_name = os.path.basename(img_train_path)
        base = os.path.splitext(img_name)[0]

        mask_train_path = os.path.join(OUTPUT_DIR, "masks/train", base + "_mask.png")
        label_train_path = os.path.join(OUTPUT_DIR, "labels/train", base + ".txt")

        img_val_path = os.path.join(OUTPUT_DIR, "images/val", img_name)
        mask_val_path = os.path.join(OUTPUT_DIR, "masks/val", base + "_mask.png")
        label_val_path = os.path.join(OUTPUT_DIR, "labels/val", base + ".txt")

        shutil.move(img_train_path, img_val_path)

        if os.path.exists(mask_train_path):
            shutil.move(mask_train_path, mask_val_path)

        if os.path.exists(label_train_path):
            shutil.move(label_train_path, label_val_path)

move_to_val(val_files)

print(f"\nDivisão: {len(train_files)} treino / {len(val_files)} validação")

data_config = {
    "train": os.path.abspath(os.path.join(OUTPUT_DIR, "images/train")),
    "val": os.path.abspath(os.path.join(OUTPUT_DIR, "images/val")),
    "nc": len(CLASSES),
    "names": list(CLASSES.keys()),
}

yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_config, f, default_flow_style=False)