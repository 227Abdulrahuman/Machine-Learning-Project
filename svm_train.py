import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

data = np.load("serialized/features.npz")

X = data["features"]
y = data["labels"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_validate, y_train, y_validate = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=7,
    stratify=y_encoded
)

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True, random_state=7))
])

svm_model.fit(X_train, y_train)

svm_preds = svm_model.predict(X_validate)
svm_acc = accuracy_score(y_validate, svm_preds)

print(f"SVM Validation Accuracy: {svm_acc * 100:.2f}%")

joblib.dump(svm_model, "serialized/svm_scaling.pkl")
joblib.dump(le, "serialized/label_encoder.pkl")
