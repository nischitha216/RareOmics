
import os
import json
import platform

import joblib
import pandas as pd
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(
    BASE_DIR, "results", "model_results"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading datasets...")

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
    genomics.index,
    "mRNA Subtype Clusters"
]

if not genomics.index.equals(epigenomics.index):
    raise ValueError("Genomics and epigenomics are not aligned.")

if not genomics.index.equals(y.index):
    raise ValueError("Features and labels are not aligned.")


# --------------------------------------------------
# 3. COMBINE OMICS FEATURES
# --------------------------------------------------

genomics.columns = [
    "GEN_" + str(c) for c in genomics.columns
]

epigenomics.columns = [
    "METH_" + str(c) for c in epigenomics.columns
]

X = pd.concat(
    [genomics, epigenomics],
    axis=1
)

feature_names = X.columns.tolist()

print("\nDataset shape:", X.shape)
print("Samples:", len(X))
print("Genomic features:", genomics.shape[1])
print("Epigenomic features:", epigenomics.shape[1])

print("\nSubtype distribution:")
print(y.value_counts())


# --------------------------------------------------
# 4. CREATE MODEL PIPELINE
# --------------------------------------------------

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


# --------------------------------------------------
# 5. DEFINE BOTH MODELS
# --------------------------------------------------

models = {

    "logistic_regression": create_pipeline(
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "random_forest": create_pipeline(
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
# 6. TRAIN AND SAVE BOTH MODELS
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("TRAINING:", model_name.upper())
    print("=" * 55)

    model.fit(X, y)

    model_path = os.path.join(
        MODEL_DIR,
        f"integrated_{model_name}.joblib"
    )

    joblib.dump(model, model_path)

    print("Training completed.")
    print("Saved model:", model_path)

    print(
        "Selected features:",
        model.named_steps["feature_selection"]
        .get_support().sum()
    )

    print(
        "Classes:",
        model.named_steps["classifier"].classes_.tolist()
    )

    # Reload model to verify it was saved correctly
    loaded_model = joblib.load(model_path)

    predictions = loaded_model.predict(X.iloc[:5])
    probabilities = loaded_model.predict_proba(X.iloc[:5])

    assert len(predictions) == 5
    assert probabilities.shape == (
        5,
        len(loaded_model.classes_)
    )

    print("Test predictions:", predictions)
    print("Probability shape:", probabilities.shape)
    print("Model verification: PASSED")


# --------------------------------------------------
# 7. SAVE FEATURE NAMES
# --------------------------------------------------

feature_path = os.path.join(
    MODEL_DIR,
    "integrated_model_feature_names.json"
)

with open(feature_path, "w") as file:
    json.dump(feature_names, file, indent=4)


# --------------------------------------------------
# 8. SAVE METADATA
# --------------------------------------------------

metadata = {

    "project": "RareTumor-Omics",

    "training_samples": int(X.shape[0]),

    "total_input_features": int(X.shape[1]),

    "genomic_features": int(genomics.shape[1]),

    "epigenomic_features": int(epigenomics.shape[1]),

    "selected_features": 100,

    "input_omics": [
        "Genomics",
        "Epigenomics"
    ],

    "target": "mRNA Subtype Clusters",

    "classes": [
        str(c) for c in y.unique()
    ],

    "models": {
        "logistic_regression": {
            "file": "integrated_logistic_regression.joblib",
            "cross_validation_accuracy": 0.861008,
            "cross_validation_macro_f1": 0.798833,
            "cross_validation_balanced_accuracy": 0.794093
        },

        "random_forest": {
            "file": "integrated_random_forest.joblib",
            "cross_validation_accuracy": 0.849412,
            "cross_validation_macro_f1": 0.759935,
            "cross_validation_balanced_accuracy": 0.757857
        }
    },

    "python_version": platform.python_version(),

    "sklearn_version": sklearn.__version__,

    "note": (
        "Both pipelines were fitted on all labelled samples. "
        "Cross-validation metrics are evaluation results, "
        "not training accuracy or external validation."
    )
}

metadata_path = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)


# --------------------------------------------------
# 9. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINAL TRAINING COMPLETED")
print("=" * 55)

print("\nSaved files:")

print("1. integrated_logistic_regression.joblib")
print("2. integrated_random_forest.joblib")
print("3. integrated_model_feature_names.json")
print("4. model_metadata.json")

print("\nBoth trained models were successfully verified.")
