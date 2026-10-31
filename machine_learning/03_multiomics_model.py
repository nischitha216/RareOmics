
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

genomics = pd.read_csv(
    os.path.join(BASE_DIR, "Genomics_Subtype_Aligned.csv"),
    index_col=0
)

epigenomics = pd.read_csv(
    os.path.join(BASE_DIR, "Epigenomics_Subtype_Aligned.csv"),
    index_col=0
)

labels = pd.read_csv(
    os.path.join(BASE_DIR, "Clinical_Subtype_Labels.csv")
)

labels["Patient_ID"] = (
    labels["Sample ID"].astype(str)
    .str.replace(".", "-", regex=False)
    .str[:12]
)

y = labels.set_index("Patient_ID").loc[
    genomics.index, "mRNA Subtype Clusters"
]

# Verify exact sample alignment
if not genomics.index.equals(epigenomics.index):
    raise ValueError("Genomics and epigenomics samples are not aligned.")

if not genomics.index.equals(y.index):
    raise ValueError("Features and subtype labels are not aligned.")

# Add prefixes to distinguish feature sources
genomics.columns = ["GEN_" + str(c) for c in genomics.columns]
epigenomics.columns = ["METH_" + str(c) for c in epigenomics.columns]

# Combine both omics datasets
X = pd.concat([genomics, epigenomics], axis=1)

print("Integrated dataset shape:", X.shape)
print("Genomics features:", genomics.shape[1])
print("Epigenomics features:", epigenomics.shape[1])

print("\nSubtype distribution:")
print(y.value_counts())

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

# All imputation, filtering, selection and scaling
# are fitted separately inside each training fold.

def create_pipeline(classifier):

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance_filter", VarianceThreshold()),
        ("feature_selection", SelectKBest(
            score_func=f_classif,
            k=100
        )),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])


models = {
    "Logistic Regression": create_pipeline(
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Random Forest": create_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        )
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

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=-1
    )

    print("\nClassification Report:")
    print(classification_report(y, predictions, zero_division=0))

    safe_name = model_name.lower().replace(" ", "_")

    prediction_df = pd.DataFrame({
        "Patient_ID": X.index,
        "Actual_Subtype": y.values,
        "Predicted_Subtype": predictions
    })

    prediction_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"multiomics_{safe_name}_predictions.csv"
        ),
        index=False
    )

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
        os.path.join(
            TABLES_DIR,
            f"multiomics_{safe_name}_confusion_matrix.csv"
        )
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens")
    plt.title(f"{model_name} - Integrated Multi-omics Confusion Matrix")
    plt.xlabel("Predicted Subtype")
    plt.ylabel("Actual Subtype")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"multiomics_{safe_name}_confusion_matrix.png"
        ),
        dpi=300
    )
    plt.close()

# --------------------------------------------------
# 5. SAVE COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(RESULTS_DIR, "Multiomics_Model_Comparison.csv"),
    index=False
)

print("\n" + "=" * 55)
print("INTEGRATED MULTI-OMICS MODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))

print("\nIntegrated multi-omics classification completed.")
