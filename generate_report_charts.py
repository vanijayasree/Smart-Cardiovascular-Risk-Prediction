"""
generate_report_charts.py

Standalone script for generating model comparison charts for your project
report / seminar presentation. This does NOT touch your live app's
models/heart_model.pkl or models/scaler.pkl — it saves everything to a
separate 'report_charts' folder instead.

Run it with:
    python generate_report_charts.py

All charts are saved as PNG files in the 'report_charts' folder — no popup
windows, no need to close anything, just run and wait.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # no popup windows — saves straight to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "report_charts")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: report_charts/{name}")


def plot_cm_with_report(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["No Disease", "Heart Disease"],
        output_dict=True
    )

    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(f"Results — {model_name}", fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    labels = ["No Disease", "Heart Disease"]

    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks(range(2)); ax1.set_xticklabels(labels, rotation=10)
    ax1.set_yticks(range(2)); ax1.set_yticklabels(labels)
    ax1.set_xlabel("Predicted", fontsize=11)
    ax1.set_ylabel("Actual", fontsize=11)
    ax1.set_title(f"Confusion Matrix [{model_name}]", fontsize=11)

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax1.text(j, i, f'{cm[i, j]:,}',
                     ha='center', va='center',
                     color=color, fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    row_labels  = ["No Disease", "Heart Disease", "", "Macro Avg", "Weighted Avg"]
    col_labels  = ["Precision", "Recall", "F1-score", "Support"]

    table_data = []
    for lbl in ["No Disease", "Heart Disease"]:
        r = report[lbl]
        table_data.append([
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1-score']:.4f}",
            f"{int(r['support']):,}"
        ])
    table_data.append(["", "", "", ""])
    for key in ["macro avg", "weighted avg"]:
        r = report[key]
        table_data.append([
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1-score']:.4f}",
            f"{int(r['support']):,}"
        ])

    tbl = ax2.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.7)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#185FA5')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    row_colors = ['#FFF3E0', '#E3F2FD']
    for i in range(2):
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(row_colors[i])

    for i in [4, 5]:
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor('#F5F5F5')

    ax2.set_title(f"Classification Report [{model_name}]", fontsize=11, pad=10)

    acc = report['accuracy']
    mf1 = report['macro avg']['f1-score']
    fig.text(
        0.73, -0.03,
        f"Accuracy: {acc:.4f}   |   Macro F1: {mf1:.4f}",
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#185FA5')
    )

    plt.tight_layout()
    savefig(f"results_{model_name.replace(' ', '_')}.png")


print("=" * 50)
print("Generating report charts — this will take a minute...")
print("=" * 50)

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv(os.path.join(BASE_DIR, "heart_clean.csv"))

X = df.drop(["S.NO", "HeartDisease"], axis=1)
y = df["HeartDisease"]

print("\nNumber of features used:", X.shape[1])

# ===============================
# 2. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 3. Cross Validation (on Random Forest)
# ===============================
rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_cv, X, y, cv=5, scoring='f1')
print("\nCross Validation F1 Scores:", cv_scores)
print("Average CV F1 Score:", cv_scores.mean())

# ===============================
# 4. Define Models
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
}

best_model, best_model_name, best_accuracy = None, "", 0
model_names, train_scores, test_scores = [], [], []
precision_scores, recall_scores, f1_scores = [], [], []

print("\nModel Comparison Results:")

# ===============================
# 5. Train, Evaluate & Track Best Model
# ===============================
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    train_acc = model.score(X_train, y_train)

    print(f"\n{name} Results:")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Test Accuracy  : {accuracy:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1 Score       : {f1:.4f}")

    plot_cm_with_report(y_test, y_pred, name)

    model_names.append(name)
    train_scores.append(train_acc)
    test_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
        best_y_pred = y_pred
        best_accuracy_val, best_precision_val = accuracy, precision
        best_recall_val, best_f1_val = recall, f1

print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

# ===============================
# 6. Train vs Test Accuracy Bar Chart
# ===============================
x = np.arange(len(model_names))
plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - 0.2, train_scores, width=0.4, label='Train')
bars2 = plt.bar(x + 0.2, test_scores,  width=0.4, label='Test')
plt.xticks(x, model_names)
plt.title("Train vs Test Accuracy - All Models")
plt.ylabel("Accuracy")
plt.legend()
for bar in bars1:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.2f}', ha='center', va='bottom')
for bar in bars2:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.2f}', ha='center', va='bottom')
plt.tight_layout()
savefig("train_vs_test_accuracy.png")

# ===============================
# 7. Model Metrics Comparison Chart
# ===============================
metrics_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(metrics_labels))
width = 0.25

lr_vals = [test_scores[0], precision_scores[0], recall_scores[0], f1_scores[0]]
dt_vals = [test_scores[1], precision_scores[1], recall_scores[1], f1_scores[1]]
rf_vals = [test_scores[2], precision_scores[2], recall_scores[2], f1_scores[2]]

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width, lr_vals, width, label='Logistic Regression', color='steelblue')
b2 = ax.bar(x,         dt_vals, width, label='Decision Tree',       color='seagreen')
b3 = ax.bar(x + width, rf_vals, width, label='Random Forest',       color='coral')
for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics")
ax.legend()
plt.tight_layout()
savefig("model_metrics_comparison.png")

# ===============================
# 8. Best Model — Train vs Test Accuracy
# ===============================
train_acc_best = best_model.score(X_train, y_train)
test_acc_best  = best_model.score(X_test,  y_test)
plt.figure()
bars = plt.bar(["Train Accuracy", "Test Accuracy"], [train_acc_best, test_acc_best],
               color=['steelblue', 'coral'])
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.4f}', ha='center', va='bottom')
plt.title(f"Train vs Test Accuracy - {best_model_name}")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.tight_layout()
savefig("best_model_train_vs_test.png")

# ===============================
# 9. Best Model — Performance Metrics Bar Chart
# ===============================
metrics = ["Accuracy", "Precision", "Recall", "F1"]
values  = [best_accuracy_val, best_precision_val, best_recall_val, best_f1_val]
plt.figure()
bars = plt.bar(metrics, values, color=['steelblue', 'seagreen', 'orange', 'purple'])
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.4f}', ha='center', va='bottom')
plt.title(f"Model Performance Metrics - {best_model_name}")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.tight_layout()
savefig("best_model_metrics.png")

# ===============================
# 10. Confusion Matrix — Best Model (standalone)
# ===============================
cm_best = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name} (Best)")
plt.tight_layout()
savefig("best_model_confusion_matrix.png")

# ===============================
# 11. ROC Curve — Best Model
# ===============================
y_probs = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_model_name}")
plt.legend()
plt.tight_layout()
savefig("best_model_roc_curve.png")

# ===============================
# 12. Feature Importance (Random Forest Only)
# ===============================
if type(best_model).__name__ == "RandomForestClassifier":
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df, hue="Feature", legend=False)
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    savefig("feature_importance.png")

print("\n" + "=" * 50)
print(f"DONE! All charts saved in: {OUT_DIR}")
print("These are for your report/presentation only —")
print("your live app's model files were NOT touched.")
print("=" * 50)
