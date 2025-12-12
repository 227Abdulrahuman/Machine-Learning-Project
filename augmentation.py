import cv2, os, random, shutil
from pathlib import Path
import numpy as np

script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

raw_dir = 'dataset/raw'
processed_dir = 'dataset/processed'
data_increase = 0.50


def augment_image(image):
    method = random.choice(['flip', 'rotate', 'brightness'])

    if method == 'flip':
        return cv2.flip(image, 1)
    elif method == 'rotate':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif method == 'brightness':
        hsb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsb = np.array(hsb, dtype=np.float64)
        brightness_ration = random.uniform(0.5, 1.4)
        hsb[:, :, 2] = hsb[:, :, 2] * brightness_ration
        hsb[:, :, 2] = np.clip(hsb[:, :, 2], 0, 255)
        hsb = np.array(hsb, dtype=np.uint8)
        return cv2.cvtColor(hsb, cv2.COLOR_HSV2BGR)


classes = [d for d in os.listdir(raw_dir) ]

for class_name in classes:

    class_src_dir = os.path.join(raw_dir, class_name)
    class_dst_dir = os.path.join(processed_dir, class_name)

    os.makedirs(class_dst_dir, exist_ok=True)

    all_files = os.listdir(class_src_dir)

    valid_images = []

    for f in all_files:
        src_path = os.path.join(class_src_dir, f)
        img = cv2.imread(src_path)

        if img is not None:
            valid_images.append(f)
            cv2.imwrite(os.path.join(class_dst_dir, f), img)

    current_count = len(valid_images)

    target_new = int(current_count * data_increase)

    count = 0
    while count < target_new:
        random_img_name = random.choice(valid_images)
        src_path = os.path.join(class_src_dir, random_img_name)

        img = cv2.imread(src_path)

        if img is None: continue

        try:
            aug_img = augment_image(img)
            new_name = f"aug_{count}_{random_img_name}"
            cv2.imwrite(os.path.join(class_dst_dir, new_name), aug_img)
            count += 1
        except:
            pass
