import os
import cv2
import albumentations as A
from glob import glob

ROOT = r"dataset"
AUG_PER_CASE = 12
OUTPUT = r"dataset_augmented"

os.makedirs(OUTPUT, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                       rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.RandomBrightnessContrast(p=0.3),
])

for class_name in ["benign", "malignant", "normal"]:
    if class_name == "normal":
        AUG_PER_CASE += 8

    in_dir = os.path.join(ROOT, class_name)
    out_dir = os.path.join(OUTPUT, class_name)
    os.makedirs(out_dir, exist_ok=True)

    cases = sorted(glob(os.path.join(in_dir, "*.png")))
    cases = [c for c in cases if "_mask" not in c]

    for img_path in cases:
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(in_dir, f"{base}_mask.png")

        if not os.path.exists(mask_path):
            print("Máscara não encontrada para:", img_path)
            continue

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(os.path.join(out_dir, base + ".png"), image)
        cv2.imwrite(os.path.join(out_dir, base + "_mask.png"), mask)

        for i in range(AUG_PER_CASE):
            augmented = transform(image=image, mask=mask)
            
            aug_img = augmented["image"]
            aug_mask = augmented["mask"]

            out_img = os.path.join(out_dir, f"{base}_aug{i}.png")
            out_mask = os.path.join(out_dir, f"{base}_aug{i}_mask.png")

            cv2.imwrite(out_img, aug_img)
            cv2.imwrite(out_mask, aug_mask)
