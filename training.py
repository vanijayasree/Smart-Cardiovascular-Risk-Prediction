import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# Load Dataset
# --------------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "heart_clean.csv"))

print("Available columns:", df.columns)

# Drop S.NO if it exists (since it's just an index column)
if "S.NO" in df.columns:
    df = df.drop("S.NO", axis=1)

# Split features (X) and target (y)
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# --------------------
# Train-Test Split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------
# Models
# --------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    results[name] = {
        "Accuracy": score,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

    # Save the best model (Random Forest usually best)
    if score > best_score:
        best_score = score
        best_model = model

# --------------------
# Convert results to DataFrame
# --------------------
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Results:")
print(results_df)

# --------------------
# Visualization: Heatmap of metrics
# --------------------
plt.figure(figsize=(8, 5))
sns.heatmap(results_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Model Evaluation Metrics")
plt.show()

# --------------------
# Feature Importance (Random Forest only)
# --------------------
if isinstance(best_model, RandomForestClassifier):
    feature_importances = best_model.feature_importances_
    features = X.columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance (Random Forest):")
    print(importance_df)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Feature Importance from Random Forest")
    plt.show()
