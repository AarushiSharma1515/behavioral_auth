import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score
)

# ── Load saved models and test data ───────────────────────
models    = joblib.load("data/models.pkl")
scaler    = joblib.load("data/scaler.pkl")
X_test, y_test = joblib.load("data/test_data.pkl")

print("Models loaded!")
print("Test samples:", len(y_test))

# ── PLOT 1: Confusion Matrices ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

colors = ["Purples", "Greens", "Oranges"]

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    acc    = accuracy_score(y_test, y_pred)

    ax = axes[idx]
    ax.imshow(cm, cmap=colors[idx])
    ax.set_title(f"{name}\nAccuracy: {acc:.3f}", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Impostor", "Authorized"])
    ax.set_yticklabels(["Impostor", "Authorized"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white", fontsize=20, fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Confusion matrices saved!")

# ── PLOT 2: ROC Curves ─────────────────────────────────────
plt.figure(figsize=(8, 6))
colors = ["#7c3aed", "#059669", "#ea580c"]

for idx, (name, model) in enumerate(models.items()):
    y_proba      = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _  = roc_curve(y_test, y_proba)
    roc_auc      = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[idx], lw=2,
             label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("ROC curves saved!")

# ── PLOT 3: Feature Importance (Random Forest) ─────────────
df_features = pd.read_csv("data/features.csv")
feature_names = df_features.drop("label", axis=1).columns.tolist()

rf_model    = models["Random Forest"]
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)),
        importances[sorted_idx],
        color="#7c3aed", alpha=0.8)
plt.xticks(range(len(feature_names)),
           [feature_names[i] for i in sorted_idx],
           rotation=45, ha="right")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Feature importance saved!")

print("\nAll plots generated successfully!")
print("Files: confusion_matrices.png, roc_curves.png, feature_importance.png")