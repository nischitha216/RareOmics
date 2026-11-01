
import json
import joblib
import pandas as pd

from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "results" / "model_results"

LR_MODEL_PATH = MODEL_DIR / "integrated_logistic_regression.joblib"
RF_MODEL_PATH = MODEL_DIR / "integrated_random_forest.joblib"

FEATURE_PATH = MODEL_DIR / "integrated_model_feature_names.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# --------------------------------------------------
# 2. LOAD MODEL METADATA
# --------------------------------------------------

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

with open(FEATURE_PATH, "r") as file:
    expected_features = json.load(file)


# --------------------------------------------------
# 3. LOAD TRAINED MODELS
# --------------------------------------------------

logistic_model = joblib.load(LR_MODEL_PATH)
random_forest_model = joblib.load(RF_MODEL_PATH)


# --------------------------------------------------
# 4. VALIDATE MODEL INPUT REQUIREMENTS
# --------------------------------------------------

EXPECTED_FEATURE_COUNT = metadata["total_input_features"]

if len(expected_features) != EXPECTED_FEATURE_COUNT:
    raise ValueError("Feature count does not match model metadata.")

if logistic_model.n_features_in_ != EXPECTED_FEATURE_COUNT:
    raise ValueError("Logistic Regression input feature mismatch.")

if random_forest_model.n_features_in_ != EXPECTED_FEATURE_COUNT:
    raise ValueError("Random Forest input feature mismatch.")


# --------------------------------------------------
# 5. PREDICTION FUNCTION
# --------------------------------------------------

def predict_subtype(data: pd.DataFrame, model_name: str = "logistic_regression"):

    # Validate required features
    missing_features = set(expected_features) - set(data.columns)
    extra_features = set(data.columns) - set(expected_features)

    if missing_features:
        raise ValueError(
            f"Missing features: {len(missing_features)}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected features: {len(extra_features)}"
        )

    # Arrange features in the exact training order
    data = data[expected_features]

    # Select model
    if model_name == "logistic_regression":
        model = logistic_model

    elif model_name == "random_forest":
        model = random_forest_model

    else:
        raise ValueError("Unsupported model selected.")

    # Generate predictions
    predictions = model.predict(data)

    probabilities = model.predict_proba(data)

    # Prepare output
    results = []

    for index, prediction in enumerate(predictions):

        results.append({
            "predicted_subtype": str(prediction),
            "probabilities": {
                str(class_name): float(probabilities[index][i])
                for i, class_name in enumerate(model.classes_)
            }
        })

    return results
