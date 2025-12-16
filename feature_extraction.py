import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from pathlib import Path

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(250, 250, 3)
)
#1280 feature.
output = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(base_model.input, output)

data_dir = Path("dataset/processed")

features = []
labels = []

batch_size = 32
image_batch = []
label_batch = []

for category in data_dir.iterdir():
    for image_path in category.iterdir():
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_batch.append(img)
        label_batch.append(category.stem)

        if len(image_batch) == batch_size:
            batch = preprocess_input(np.array(image_batch))
            batch_features = feature_extractor(batch, training=False).numpy()

            features.append(batch_features)
            labels.extend(label_batch)

            image_batch.clear()
            label_batch.clear()

if image_batch:
    batch = preprocess_input(np.array(image_batch))
    batch_features = feature_extractor(batch, training=False).numpy() #Make extractor deterministic.
    features.append(batch_features)
    labels.extend(label_batch)

features = np.vstack(features)
labels = np.array(labels)

np.savez_compressed("serialized/features.npz", features=features, labels=labels)
