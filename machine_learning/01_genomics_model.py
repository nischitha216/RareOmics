
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score,
    balanced_accuracy_score
)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results", "model_results")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

X_path = os.path.join(BASE_DIR, "Genomics_Subtype_Aligned.csv")
y_path = os.path.join(BASE_DIR, "Clinical_Subtype_Labels.csv")

X = pd.read_csv(X_path, index_col=0)
labels = pd.read_csv(y_path)

y = labels.set_index("Patient_ID").loc[X.index, "mRNA Subtype Clusters"]

print("Genomics dataset shape:", X.shape)
print("\nSubtype distribution:")
print(y.value_counts())

# Check data
if X.isna().sum().sum() > 0:
    raise ValueError("Missing values found in genomics dataset.")

if not X.index.equals(y.index):
    raise ValueError("Sample IDs are not aligned.")

# --------------------------------------------------
# 3. CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy"
}

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# --------------------------------------------------
# 4. TRAIN AND EVALUATE
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    accuracy = cv_results["test_accuracy"]
    macro_f1 = cv_results["test_macro_f1"]
    balanced_acc = cv_results["test_balanced_accuracy"]

    print("Mean Accuracy:", round(accuracy.mean(), 4))
    print("Mean Macro-F1:", round(macro_f1.mean(), 4))
    print("Mean Balanced Accuracy:", round(balanced_acc.mean(), 4))

    results.append({
        "Model": model_name,
        "Accuracy_Mean": accuracy.mean(),
        "Accuracy_STD": accuracy.std(),
        "Macro_F1_Mean": macro_f1.mean(),
        "Macro_F1_STD": macro_f1.std(),
        "Balanced_Accuracy_Mean": balanced_acc.mean(),
        "Balanced_Accuracy_STD": balanced_acc.std()
    })

    # Out-of-fold predictions: every sample is predicted
    # by a model that did not train on that sample.
    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=-1
    )

    print("\nClassification Report:")
    print(classification_report(y, predictions, zero_division=0))

    # Save predictions
    prediction_df = pd.DataFrame({
        "Patient_ID": X.index,
        "Actual_Subtype": y.values,
        "Predicted_Subtype": predictions
    })

    safe_name = model_name.lower().replace(" ", "_")

    prediction_df.to_csv(
        os.path.join(RESULTS_DIR, f"{safe_name}_predictions.csv"),
        index=False
    )

    # Confusion matrix
    class_names = sorted(y.unique())

    cm = confusion_matrix(
        y,
        predictions,
        labels=class_names
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.to_csv(
        os.path.join(TABLES_DIR, f"{safe_name}_confusion_matrix.csv")
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title(f"{model_name} - Genomics Confusion Matrix")
    plt.xlabel("Predicted Subtype")
    plt.ylabel("Actual Subtype")
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_DIR, f"{safe_name}_confusion_matrix.png"),
        dpi=300
    )
    plt.close()

# --------------------------------------------------
# 5. SAVE MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(RESULTS_DIR, "Genomics_Model_Comparison.csv"),
    index=False
)

print("\n" + "=" * 55)
print("GENOMICS MODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))

print("\nGenomics classification completed.")
