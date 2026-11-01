
import argparse
import json
import os
import sys

import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "results", "model_results")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "integrated_logistic_regression.joblib"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "integrated_model_feature_names.json"
)


def load_model_and_features():
    """Load the trained model and its expected input feature names."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(f"Feature list not found: {FEATURE_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH, "r") as file:
        expected_features = json.load(file)

    return model, expected_features


def prepare_input(genomics_path, epigenomics_path, expected_features):
    """Prepare input CSV files in exactly the same format as training."""

    genomics = pd.read_csv(genomics_path, index_col=0)
    epigenomics = pd.read_csv(epigenomics_path, index_col=0)

    if genomics.empty or epigenomics.empty:
        raise ValueError("One or both input datasets are empty.")

    if not genomics.index.equals(epigenomics.index):
        raise ValueError(
            "Genomics and epigenomics must contain the same samples "
            "in the same order."
        )

    if genomics.index.has_duplicates or epigenomics.index.has_duplicates:
        raise ValueError("Duplicate sample IDs found.")

    if genomics.columns.has_duplicates or epigenomics.columns.has_duplicates:
        raise ValueError("Duplicate feature names found.")

    # Add the same prefixes used during model training.
    genomics.columns = ["GEN_" + str(c) for c in genomics.columns]
    epigenomics.columns = ["METH_" + str(c) for c in epigenomics.columns]

    # Combine the two omics datasets.
    X = pd.concat([genomics, epigenomics], axis=1)

    if X.columns.has_duplicates:
        raise ValueError("Duplicate feature names after combining datasets.")

    # Check that every feature required by the model is present.
    missing_features = [
        feature for feature in expected_features
        if feature not in X.columns
    ]

    if missing_features:
        preview = ", ".join(missing_features[:10])

        raise ValueError(
            f"{len(missing_features)} required features are missing. "
            f"Examples: {preview}. "
            "Check the input feature names and preprocessing."
        )

    # Keep the exact feature order used during training.
    X = X.reindex(columns=expected_features)

    # Ensure all values are numeric.
    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("Input contains non-numeric feature values.") from error

    return X


def predict_subtypes(genomics_path, epigenomics_path, output_path):
    """Predict tumour subtypes and save class probabilities."""

    model, expected_features = load_model_and_features()

    X = prepare_input(
        genomics_path,
        epigenomics_path,
        expected_features
    )

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    classes = model.named_steps["classifier"].classes_

    results = pd.DataFrame({
        "Sample_ID": X.index,
        "Predicted_Subtype": predictions
    })

    for index, class_name in enumerate(classes):
        results[f"Probability_{class_name}"] = probabilities[:, index]

    results.to_csv(output_path, index=False)

    print("\nPrediction completed successfully.")
    print("Samples processed:", len(X))
    print("Features supplied:", X.shape[1])
    print("Model: Integrated Logistic Regression")
    print("\nPredictions:")
    print(results.to_string(index=False))
    print("\nResults saved to:", output_path)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict PCPG molecular subtype using integrated omics data."
    )

    parser.add_argument(
        "--genomics",
        required=True,
        help="Path to processed genomic CSV file."
    )

    parser.add_argument(
        "--epigenomics",
        required=True,
        help="Path to processed epigenomic CSV file."
    )

    parser.add_argument(
        "--output",
        default="subtype_predictions.csv",
        help="Output CSV filename."
    )

    args = parser.parse_args()

    try:
        predict_subtypes(
            args.genomics,
            args.epigenomics,
            args.output
        )

    except Exception as error:
        print("\nPrediction failed:", error)
        sys.exit(1)
