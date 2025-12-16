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

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(250, 250, 3))
output = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(base_model.input, output)

svm_model = joblib.load("serialized/svm_scaling.pkl")
knn_model = joblib.load("serialized/knn_scaling.pkl")
le = joblib.load("serialized/label_encoder.pkl")



def test_image_svm(image_path, threshold=0.8):
    image = cv2.imread(image_path)
    if image is None:
        print("Image Not Found")
        return

    image = cv2.resize(image, (250, 250))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = feature_extractor.predict(image, verbose=0)
    probabilities = svm_model.predict_proba(features)[0]

    svm_probability = np.max(probabilities)
    label = np.argmax(probabilities)

    svm_pred = le.inverse_transform([label])[0]

    if svm_probability <= threshold:
        svm_pred = "Unknown"

    print(f"SVM Prediction: {svm_pred} (Probability: {svm_probability * 100:.2f}%)")

def test_image_knn(image_path, threshold=0.7):
    image = cv2.imread(image_path)
    if image is None:
        print("Image Not Found")
        return

    image = cv2.resize(image, (250, 250))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = feature_extractor.predict(image, verbose=0)
    probabilities = knn_model.predict_proba(features)[0]

    knn_probability = np.max(probabilities)
    label = np.argmax(probabilities)

    knn_pred = le.inverse_transform([label])[0]

    if knn_probability <= threshold:
        knn_pred = "Unknown"

    print(f"KNN Prediction: {knn_pred} (Probability: {knn_probability * 100:.2f}%)")




test_image_svm('dataset/tests/glass.jpg')
test_image_svm('dataset/tests/cardboard.jpg')
test_image_svm('dataset/tests/metal.jpg')
test_image_svm('dataset/tests/paper.jpg')
test_image_svm('dataset/tests/unknown.jpg')
print()
print()
test_image_knn('dataset/tests/glass.jpg')
test_image_knn('dataset/tests/cardboard.jpg')
test_image_knn('dataset/tests/metal.jpg')
test_image_knn('dataset/tests/paper.jpg')
test_image_knn('dataset/tests/unknown.jpg')



