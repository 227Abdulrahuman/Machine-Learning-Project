import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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


knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski"
    ))
])

knn_model.fit(X_train, y_train)

knn_preds = knn_model.predict(X_validate)
knn_acc = accuracy_score(y_validate, knn_preds)

print(f"KNN Validation Accuracy: {knn_acc * 100:.2f}%")

joblib.dump(knn_model, "serialized/knn_scaling.pkl")
