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

svm_threshold = 0.6
knn_threshold = 0.7

cap = cv2.VideoCapture(0)
roi_top_left = (100, 100)
roi_bottom_right = (400, 400)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 0, 0), 2)
    cv2.putText(frame, "Place object in box", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    x1, y1 = roi_top_left
    x2, y2 = roi_bottom_right
    roi = frame[y1:y2, x1:x2]

    img = cv2.resize(roi, (250, 250))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = feature_extractor.predict(img, verbose=0)

    svm_probs = svm_model.predict_proba(features)[0]
    svm_idx = np.argmax(svm_probs)
    svm_prob = np.max(svm_probs)
    svm_label = le.inverse_transform([svm_idx])[0] if svm_prob > svm_threshold else "Unknown"

    knn_probs = knn_model.predict_proba(features)[0]
    knn_idx = np.argmax(knn_probs)
    knn_prob = np.max(knn_probs)
    knn_label = le.inverse_transform([knn_idx])[0] if knn_prob > knn_threshold else "Unknown"

    predicted_text = f"SVM: {svm_label} ({svm_prob*100:.1f}%) | KNN: {knn_label} ({knn_prob*100:.1f}%)"
    color = (0, 0, 255) if "Unknown" in predicted_text else (0, 255, 0)
    cv2.putText(frame, predicted_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    cv2.imshow("Realtime Object Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
