
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Literal
import pandas as pd
import numpy as np
from io import BytesIO

from app.services.model_service import (
    predict_subtype,
    expected_features,
    metadata
)

router = APIRouter()

# Separate expected features by omics type
GENOMIC_FEATURES = [
    feature for feature in expected_features
    if feature.startswith("GEN_")
]

METHYLATION_FEATURES = [
    feature for feature in expected_features
    if feature.startswith("METH_")
]


def read_csv_file(file_content, filename):
    """Read an uploaded CSV using its first column as sample IDs."""

    try:
        df = pd.read_csv(BytesIO(file_content), index_col=0)

        if df.empty:
            raise ValueError(f"{filename} is empty.")

        if df.index.isna().any():
            raise ValueError(f"{filename} contains missing sample IDs.")

        # Clean sample IDs before checking duplicates
        df.index = df.index.astype(str).str.strip()

        if (df.index == "").any():
            raise ValueError(f"{filename} contains empty sample IDs.")

        if df.index.duplicated().any():
            raise ValueError(f"{filename} contains duplicate sample IDs.")

        if df.columns.duplicated().any():
            raise ValueError(f"{filename} contains duplicate feature names.")

        # Convert feature values to numeric
        try:
            df = df.apply(pd.to_numeric, errors="raise")
        except (ValueError, TypeError):
            raise ValueError(
                f"{filename} contains non-numeric feature values."
            )

        # Reject infinite values; NaN is allowed for model imputation
        if np.isinf(df.to_numpy(dtype=float)).any():
            raise ValueError(
                f"{filename} contains infinite values."
            )

        return df

    except pd.errors.ParserError:
        raise ValueError(
            f"Could not parse {filename} as a CSV file."
        )


def validate_features(df, expected, filename):
    """Check that all expected features are present."""

    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)

    if missing or extra:
        raise ValueError(
            f"{filename} feature mismatch. "
            f"Missing: {len(missing)}, Extra: {len(extra)}."
        )

    return df[expected]


@router.post("/predict/multiomics")
async def predict_multiomics(
    genomics_file: UploadFile = File(...),
    epigenomics_file: UploadFile = File(...),
    model_name: Literal[
        "logistic_regression",
        "random_forest"
    ] = Query(default="logistic_regression")
):

    try:
        # 1. Validate file extensions
        if not genomics_file.filename.lower().endswith(".csv"):
            raise ValueError(
                "Genomics input must be a CSV file."
            )

        if not epigenomics_file.filename.lower().endswith(".csv"):
            raise ValueError(
                "Epigenomics input must be a CSV file."
            )

        # 2. Read uploaded CSV files
        genomics = read_csv_file(
            await genomics_file.read(),
            genomics_file.filename
        )

        epigenomics = read_csv_file(
            await epigenomics_file.read(),
            epigenomics_file.filename
        )

        # 3. Add prefixes expected by the trained model
        genomics.columns = [
            "GEN_" + str(col) for col in genomics.columns
        ]

        epigenomics.columns = [
            "METH_" + str(col) for col in epigenomics.columns
        ]

        # Check for duplicate columns after prefixing
        if genomics.columns.duplicated().any():
            raise ValueError(
                "Duplicate genomic features found after prefixing."
            )

        if epigenomics.columns.duplicated().any():
            raise ValueError(
                "Duplicate epigenomic features found after prefixing."
            )

        # 4. Validate feature counts against model metadata
        if len(GENOMIC_FEATURES) != metadata["genomic_features"]:
            raise ValueError(
                "Unexpected genomic feature metadata."
            )

        if len(METHYLATION_FEATURES) != metadata["epigenomic_features"]:
            raise ValueError(
                "Unexpected epigenomic feature metadata."
            )

        # 5. Validate feature names
        genomics = validate_features(
            genomics,
            GENOMIC_FEATURES,
            "Genomics"
        )

        epigenomics = validate_features(
            epigenomics,
            METHYLATION_FEATURES,
            "Epigenomics"
        )

        # 6. Validate genomic values are binary (0 or 1)
        genomic_values = genomics.to_numpy()

        valid_genomic = (
            pd.isna(genomic_values)
            | (genomic_values == 0)
            | (genomic_values == 1)
        )

        if not valid_genomic.all():
            raise ValueError(
                "Genomics features must contain only "
                "0, 1 or missing values."
            )

        # 7. Ensure both datasets contain the same samples
        if set(genomics.index) != set(epigenomics.index):
            raise ValueError(
                "Genomics and epigenomics sample IDs do not match."
            )

        # 8. Align epigenomics rows to genomics sample order
        epigenomics = epigenomics.loc[genomics.index]

        # 9. Combine both datasets
        combined = pd.concat(
            [genomics, epigenomics],
            axis=1
        )

        # Ensure exact feature order expected by the model
        combined = combined[expected_features]

        # 10. Generate predictions
        predictions = predict_subtype(
            combined,
            model_name
        )

        # 11. Attach sample IDs to prediction results
        output = []

        for patient_id, prediction in zip(
            combined.index,
            predictions
        ):
            output.append({
                "patient_id": patient_id,
                **prediction
            })

        # 12. Return API response
        return {
            "status": "success",
            "model_used": model_name,
            "samples_predicted": len(output),
            "prediction_target": metadata["target"],
            "predictions": output
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
