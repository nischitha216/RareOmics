
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENOMICS_FILE = os.path.join(BASE_DIR, "Genomics_Subtype_Aligned.csv")
EPIGENOMICS_FILE = os.path.join(BASE_DIR, "Epigenomics_Subtype_Aligned.csv")
LABEL_FILE = os.path.join(BASE_DIR, "Clinical_Subtype_Labels.csv")

RESULT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TOP_N = 20
K_FEATURES = 100

# --------------------------------------------------
# 2. LOAD AND ALIGN DATA
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

# Prefix feature names to distinguish the two omics layers.
genomics.columns = ["GEN_" + str(c) for c in genomics.columns]
epigenomics.columns = ["EPI_" + str(c) for c in epigenomics.columns]

X = pd.concat([genomics, epigenomics], axis=1)
X = X.apply(pd.to_numeric, errors="coerce")

feature_names = X.columns.to_list()
class_names = sorted(y.unique())

print("Dataset shape:", X.shape)
print("Number of samples:", len(y))
print("Number of features:", len(feature_names))
print("Classes:", class_names)

# --------------------------------------------------
# 3. STRATIFIED CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Sum of absolute SHAP values for each feature.
global_abs_sum = np.zeros(len(feature_names))
global_observation_count = np.zeros(len(feature_names))

# Signed out-of-fold SHAP values, separately for each class.
class_signed_values = {
    class_name: np.full((len(X), len(feature_names)), np.nan)
    for class_name in class_names
}

# Number of folds in which each feature was selected.
selection_count = np.zeros(len(feature_names))

# --------------------------------------------------
# 4. RUN SHAP ON HELD-OUT FOLDS
# --------------------------------------------------

print("\nStarting out-of-fold SHAP analysis...")

X_array = X.to_numpy(dtype=float)
y_array = y.to_numpy()

for fold, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array), 1):

    print(f"\nProcessing fold {fold}/{N_SPLITS}...")

    X_train = X_array[train_idx]
    X_test = X_array[test_idx]

    y_train = y_array[train_idx]

    # Imputation is fitted only on the training fold.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Remove constant features using training data only.
    variance_filter = VarianceThreshold(threshold=0)
    X_train_var = variance_filter.fit_transform(X_train_imp)
    X_test_var = variance_filter.transform(X_test_imp)

    variance_mask = variance_filter.get_support()
    variance_feature_names = np.array(feature_names)[variance_mask]

    # Select the top features using training data only.
    selector = SelectKBest(
        score_func=f_classif,
        k=min(K_FEATURES, X_train_var.shape[1])
    )

    X_train_selected = selector.fit_transform(X_train_var, y_train)
    X_test_selected = selector.transform(X_test_var)

    selected_mask = selector.get_support()
    selected_feature_names = variance_feature_names[selected_mask]

    # Record feature selection frequency.
    selected_original_indices = np.where(variance_mask)[0][selected_mask]
    selection_count[selected_original_indices] += 1

    # Scale using training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Train the integrated Logistic Regression model.
    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(X_train_scaled, y_train)

    # Explain only the held-out test samples.
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    # SHAP versions may return a list or a NumPy array.
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)
    else:
        shap_array = np.asarray(shap_values)

    # Expected shape: samples x selected features x classes.
    if shap_array.ndim == 2:
        shap_array = shap_array[:, :, np.newaxis]

    if shap_array.shape[0] != len(test_idx):
        raise ValueError(
            f"Unexpected SHAP sample dimension: {shap_array.shape}"
        )

    if shap_array.shape[1] != len(selected_feature_names):
        raise ValueError(
            f"Unexpected SHAP feature dimension: {shap_array.shape}"
        )

    if shap_array.shape[2] != len(model.classes_):
        raise ValueError(
            f"Unexpected SHAP class dimension: {shap_array.shape}"
        )

    # Map selected feature contributions back to the full feature set.
    for local_idx, original_idx in enumerate(selected_original_indices):

        values_for_feature = shap_array[:, local_idx, :]

        global_abs_sum[original_idx] += np.abs(values_for_feature).sum()
        global_observation_count[original_idx] += len(test_idx)

    # Preserve signed SHAP values for each class.
    for class_idx, class_name in enumerate(model.classes_):

        for local_idx, original_idx in enumerate(selected_original_indices):

            class_signed_values[class_name][
                test_idx, original_idx
            ] = shap_array[:, local_idx, class_idx]

    print("Selected features:", len(selected_feature_names))
    print("Fold completed.")

# --------------------------------------------------
# 5. GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

mean_abs_shap = np.divide(
    global_abs_sum,
    global_observation_count,
    out=np.zeros_like(global_abs_sum),
    where=global_observation_count > 0
)

selection_frequency = selection_count / N_SPLITS

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Omics_Type": [
        "Genomics" if name.startswith("GEN_") else "Epigenomics"
        for name in feature_names
    ],
    "Mean_Absolute_SHAP": mean_abs_shap,
    "Selection_Frequency": selection_frequency
})

importance_df = importance_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

importance_path = os.path.join(
    TABLE_DIR,
    "Integrated_SHAP_Feature_Importance.csv"
)

importance_df.to_csv(importance_path, index=False)

print("\nTop 20 globally important features:")
print(importance_df.head(TOP_N).to_string(index=False))

# --------------------------------------------------
# 6. GLOBAL SHAP BAR PLOT
# --------------------------------------------------

top_features = importance_df.head(TOP_N).sort_values(
    "Mean_Absolute_SHAP"
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"],
    top_features["Mean_Absolute_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Integrated Model: Global SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "Integrated_SHAP_Global_Importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 7. CLASS-SPECIFIC SIGNED SHAP PLOTS
# --------------------------------------------------

for class_name in class_names:

    signed_matrix = class_signed_values[class_name]

    mean_signed = np.nanmean(signed_matrix, axis=0)
    mean_absolute = np.nanmean(np.abs(signed_matrix), axis=0)

    class_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Signed_SHAP": mean_signed,
        "Mean_Absolute_SHAP": mean_absolute,
        "Selection_Frequency": selection_frequency
    })

    class_df = class_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    safe_class_name = (
        class_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    class_df.to_csv(
        os.path.join(
            TABLE_DIR,
            f"SHAP_{safe_class_name}_Feature_Importance.csv"
        ),
        index=False
    )

    # Plot signed mean SHAP for the most influential features.
    plot_df = class_df.head(TOP_N).sort_values("Mean_Signed_SHAP")

    plt.figure(figsize=(10, 8))

    plt.barh(
        plot_df["Feature"],
        plot_df["Mean_Signed_SHAP"]
    )

    plt.axvline(0, linewidth=1)
    plt.xlabel("Mean Signed SHAP Value")
    plt.ylabel("Feature")
    plt.title(f"SHAP Contributions: {class_name}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            f"SHAP_{safe_class_name}_Signed_Importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# --------------------------------------------------
# 8. FINISH
# --------------------------------------------------

print("\nSHAP analysis completed successfully.")

print("\nSaved files:")
print("- results/tables/Integrated_SHAP_Feature_Importance.csv")
print("- results/figures/Integrated_SHAP_Global_Importance.png")
print("- Class-specific SHAP tables and plots")
