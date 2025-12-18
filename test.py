import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from pathlib import Path

curr_dir = Path(__file__).resolve().parent
os.chdir(curr_dir)


mapping = {
    "glass" : 0,
    "paper" : 1,
    "cardboard" : 2,
    "plastic" : 3,
    "metal" : 4,
    "trash" : 5,
    "unknown" : 6,
}



def predict(dataFilePath='tests',bestModelPath='serialized/svm_scaling.pkl'):

    dataFilePath = Path(dataFilePath)

    for img_path in dataFilePath.iterdir():
        image = cv2.imread(img_path)
        if image is None:
            print("Image Not Found")
            return

        image = cv2.resize(image, (250, 250))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(250, 250, 3))
        output = GlobalAveragePooling2D()(base_model.output)
        feature_extractor = Model(base_model.input, output)

        svm_model = joblib.load("serialized/svm_scaling.pkl")
        le = joblib.load("serialized/label_encoder.pkl")

        features = feature_extractor.predict(image, verbose=0)
        probabilities = svm_model.predict_proba(features)[0]

        svm_probability = np.max(probabilities)
        label = np.argmax(probabilities)

        svm_pred = le.inverse_transform([label])[0]

        if svm_probability <= 0.6:
            svm_pred = "unknown"

        svm_pred = svm_pred.lower()
        id = mapping[svm_pred]

        print(f"Original {img_path.stem}, Prediction: {id} (Probability: {svm_probability * 100:.2f}%)")
