
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = os.getcwd()

GENOMICS_FILE = os.path.join(ROOT, "Genomics_Subtype_Aligned.csv")
EPIGENOMICS_FILE = os.path.join(ROOT, "Epigenomics_Subtype_Aligned.csv")
LABELS_FILE = os.path.join(ROOT, "Clinical_Subtype_Labels.csv")

TABLE_DIR = os.path.join(ROOT, "results", "tables")
FIGURE_DIR = os.path.join(ROOT, "results", "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading datasets...")

genomics = pd.read_csv(GENOMICS_FILE)
epigenomics = pd.read_csv(EPIGENOMICS_FILE)
labels_df = pd.read_csv(LABELS_FILE)

# Normalize patient IDs
for df in [genomics, epigenomics, labels_df]:
    df["Patient_ID"] = (
        df["Patient_ID"]
        .astype(str)
        .str.replace(".", "-", regex=False)
        .str[:12]
    )

# Align samples
common_ids = sorted(
    set(genomics["Patient_ID"])
    & set(epigenomics["Patient_ID"])
    & set(labels_df["Patient_ID"])
)

genomics = (
    genomics.drop_duplicates("Patient_ID")
    .set_index("Patient_ID")
    .loc[common_ids]
)

epigenomics = (
    epigenomics.drop_duplicates("Patient_ID")
    .set_index("Patient_ID")
    .loc[common_ids]
)

labels_df = (
    labels_df.drop_duplicates("Patient_ID")
    .set_index("Patient_ID")
    .loc[common_ids]
)

# Combine genomic and epigenomic features
genomics_features = genomics.drop(columns=["Sample ID"], errors="ignore")
epigenomics_features = epigenomics.drop(columns=["Sample ID"], errors="ignore")

genomics_features = genomics_features.add_prefix("GEN_")
epigenomics_features = epigenomics_features.add_prefix("EPI_")

X = pd.concat(
    [genomics_features, epigenomics_features],
    axis=1
)

y = labels_df["mRNA Subtype Clusters"]

print("Dataset shape:", X.shape)
print("Number of samples:", len(y))
print("Subtype distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. DEFINE THE SAME INTEGRATED MODEL
# --------------------------------------------------

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("variance_filter", VarianceThreshold()),

    ("feature_selection", SelectKBest(
        score_func=f_classif,
        k=100
    )),

    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ))
])

# --------------------------------------------------
# 4. GENERATE OUT-OF-FOLD PREDICTIONS
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nGenerating out-of-fold predictions...")

y_pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict",
    n_jobs=1
)

# --------------------------------------------------
# 5. EVALUATION METRICS
# --------------------------------------------------

accuracy = accuracy_score(y, y_pred)
macro_f1 = f1_score(y, y_pred, average="macro")
balanced_acc = balanced_accuracy_score(y, y_pred)

print("\nOVERALL MODEL PERFORMANCE")
print("-------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

print("\nCLASSIFICATION REPORT")
print("---------------------")

report = classification_report(
    y,
    y_pred,
    zero_division=0
)

print(report)

# Save classification report
report_dict = classification_report(
    y,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = os.path.join(
    TABLE_DIR,
    "Integrated_LR_Classification_Report.csv"
)

report_df.to_csv(report_path)

# --------------------------------------------------
# 6. CONFUSION MATRIX
# --------------------------------------------------

class_names = sorted(y.unique())

cm = confusion_matrix(
    y,
    y_pred,
    labels=class_names
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_path = os.path.join(
    TABLE_DIR,
    "Integrated_LR_Confusion_Matrix.csv"
)

cm_df.to_csv(cm_path)

# --------------------------------------------------
# 7. PLOT CONFUSION MATRIX
# --------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    cbar=True
)

plt.title(
    "Integrated Logistic Regression - Confusion Matrix",
    fontsize=14
)

plt.xlabel("Predicted Subtype")
plt.ylabel("Actual Subtype")

plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

figure_path = os.path.join(
    FIGURE_DIR,
    "Integrated_LR_Confusion_Matrix.png"
)

plt.savefig(figure_path, dpi=300)
plt.close()

# --------------------------------------------------
# 8. COMPLETION
# --------------------------------------------------

print("\nAnalysis completed successfully.")

print("\nSaved files:")
print(report_path)
print(cm_path)
print(figure_path)
