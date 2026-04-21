import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your personal dataset
df = pd.read_csv("data/my_features.csv")

print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# Separate X and y
X = df.drop("label", axis=1).values
y = df["label"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train all 3 models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":           SVC(kernel="rbf", probability=True, random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred,
          target_names=["Impostor", "Authorized"]))

# Save new models and scaler
joblib.dump(models, "data/my_models.pkl")
joblib.dump(scaler, "data/my_scaler.pkl")
print("\nYour personal models saved!")
print("Now run demo.py and choose mode 2 to authenticate!")