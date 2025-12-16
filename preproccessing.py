import cv2
from pathlib import Path
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import random

data_dir = Path('dataset/raw')
output_dir = Path('dataset/processed')

classes = ['glass', 'metal', 'plastic', 'paper', 'trash', 'cardboard']

for c in classes:
    os.makedirs(output_dir / c, exist_ok=True)


def resize():
    global data_dir
    for category in data_dir.iterdir():
        save_dir = output_dir / category.name
        for image_path in category.iterdir():
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            resized_image = cv2.resize(image, (250, 250))
            save_path = save_dir / f"{image_path.stem}.jpg"
            cv2.imwrite(str(save_path), resized_image)

    data_dir = output_dir


def flip(image):
    return tf.image.flip_left_right(image).numpy()


def rotate(image):
    datagen = ImageDataGenerator(rotation_range=60, fill_mode='nearest')
    img_batch = np.expand_dims(image, axis=0)
    aug_iter = datagen.flow(img_batch, batch_size=1)
    rotated_batch = next(aug_iter)
    return rotated_batch[0].astype('uint8')


def brightness(image):
    datagen = ImageDataGenerator(brightness_range=[0.4, 1.9], fill_mode='nearest')
    img_batch = np.expand_dims(image, axis=0)
    aug_iter = datagen.flow(img_batch, batch_size=1)
    bright_batch = next(aug_iter)
    return bright_batch[0].astype('uint8')


def augment_image(image):
    if random.random() < 0.5:
        image = flip(image)
    if random.random() < 0.5:
        image = rotate(image)
    if random.random() < 0.5:
        image = brightness(image)
    return image


def augment_all():
    for category in data_dir.iterdir():
        save_dir = output_dir / category.name
        for image_path in list(category.iterdir()):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            flipped = flip(image)
            cv2.imwrite(str(save_dir / f"{image_path.stem}-flipped.jpg"), flipped)

            rotated = rotate(image)
            cv2.imwrite(str(save_dir / f"{image_path.stem}-rotated.jpg"), rotated)

            bright = brightness(image)
            cv2.imwrite(str(save_dir / f"{image_path.stem}-bright.jpg"), bright)


def balance_dataset():
    counts = {c: len(list((output_dir / c).iterdir())) for c in classes}
    max_count = 5000

    print("Initial counts:", counts)

    for category in classes:
        save_dir = output_dir / category
        current_images = list(save_dir.iterdir())
        while len(current_images) < max_count:

            img_path = random.choice(current_images)
            image = cv2.imread(str(img_path))
            new_image = augment_image(image)
            save_path = save_dir / f"{img_path.stem}-extra-{len(current_images)}.jpg"
            cv2.imwrite(str(save_path), new_image)
            current_images.append(save_path)

    final_counts = {c: len(list((output_dir / c).iterdir())) for c in classes}
    print("Balanced counts:", final_counts)


resize()
augment_all()
balance_dataset()
