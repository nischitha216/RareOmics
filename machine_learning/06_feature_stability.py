
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENOMICS_FILE = os.path.join(BASE_DIR, "Genomics_Subtype_Aligned.csv")
EPIGENOMICS_FILE = os.path.join(BASE_DIR, "Epigenomics_Subtype_Aligned.csv")
LABEL_FILE = os.path.join(BASE_DIR, "Clinical_Subtype_Labels.csv")

TABLE_DIR = os.path.join(BASE_DIR, "results", "tables")
FIGURE_DIR = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading datasets...")

genomics = pd.read_csv(GENOMICS_FILE, index_col=0)
epigenomics = pd.read_csv(EPIGENOMICS_FILE, index_col=0)
labels_df = pd.read_csv(LABEL_FILE)

def normalize_ids(values):
    return (
        pd.Index(values.astype(str))
        .str.replace(".", "-", regex=False)
        .str[:12]
    )

genomics.index = normalize_ids(genomics.index)
epigenomics.index = normalize_ids(epigenomics.index)
labels_df["Patient_ID"] = normalize_ids(labels_df["Patient_ID"])

labels_df = labels_df.drop_duplicates("Patient_ID")
labels_df = labels_df.set_index("Patient_ID")

common_ids = labels_df.index.intersection(genomics.index)
common_ids = common_ids.intersection(epigenomics.index)

labels_df = labels_df.loc[common_ids]
genomics = genomics.loc[common_ids]
epigenomics = epigenomics.loc[common_ids]

y = labels_df["mRNA Subtype Clusters"].astype(str)

genomics.columns = ["GEN_" + str(c) for c in genomics.columns]
epigenomics.columns = ["EPI_" + str(c) for c in epigenomics.columns]

X = pd.concat([genomics, epigenomics], axis=1)
X = X.apply(pd.to_numeric, errors="coerce")

feature_names = X.columns.to_list()

print("Dataset shape:", X.shape)
print("Samples:", len(y))
print("Features:", len(feature_names))

# --------------------------------------------------
# 3. MODEL PIPELINE
# --------------------------------------------------

def create_pipeline():

    return Pipeline([
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
# 4. FIVE-FOLD FEATURE SELECTION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

selection_count = np.zeros(len(feature_names), dtype=int)

print("\nStarting feature stability analysis...")

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y), start=1
):

    print(f"\nProcessing fold {fold}/5...")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    model = create_pipeline()
    model.fit(X_train, y_train)

    # Get features remaining after variance filtering.
    variance_mask = (
        model.named_steps["variance_filter"]
        .get_support()
    )

    variance_features = np.array(feature_names)[variance_mask]

    # Get features selected by SelectKBest.
    selected_mask = (
        model.named_steps["feature_selection"]
        .get_support()
    )

    selected_features = variance_features[selected_mask]

    # Record selected features.
    for feature in selected_features:
        original_index = feature_names.index(feature)
        selection_count[original_index] += 1

    print("Selected features:", len(selected_features))

# --------------------------------------------------
# 5. CALCULATE STABILITY
# --------------------------------------------------

stability_df = pd.DataFrame({
    "Feature": feature_names,
    "Omics_Type": [
        "Genomics" if name.startswith("GEN_") else "Epigenomics"
        for name in feature_names
    ],
    "Selection_Count": selection_count,
    "Selection_Frequency": selection_count / 5
})

stability_df = stability_df.sort_values(
    ["Selection_Count", "Feature"],
    ascending=[False, True]
)

output_csv = os.path.join(
    TABLE_DIR,
    "Feature_Selection_Stability.csv"
)

stability_df.to_csv(output_csv, index=False)

# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

print("\nTOP 20 STABLE FEATURES")

print(
    stability_df.head(20).to_string(index=False)
)

print("\nFEATURES SELECTED IN ALL FIVE FOLDS:")

print(
    stability_df[
        stability_df["Selection_Count"] == 5
    ].to_string(index=False)
)

# --------------------------------------------------
# 7. PLOT FEATURE STABILITY
# --------------------------------------------------

top_features = stability_df.head(20).sort_values(
    "Selection_Count"
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"],
    top_features["Selection_Count"]
)

plt.xlabel("Number of Folds Selected (out of 5)")
plt.ylabel("Feature")
plt.title("Integrated Model: Feature Selection Stability")
plt.xticks([0, 1, 2, 3, 4, 5])
plt.xlim(0, 5.5)
plt.tight_layout()

output_plot = os.path.join(
    FIGURE_DIR,
    "Feature_Selection_Stability.png"
)

plt.savefig(output_plot, dpi=300, bbox_inches="tight")
plt.close()

print("\nAnalysis completed successfully.")

print("\nSaved:")
print(output_csv)
print(output_plot)
