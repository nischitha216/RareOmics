# Input Requirements for RareTumor-Omics Prediction Model

## 1. Purpose

This document describes the input format required by the integrated Logistic Regression model developed for the RareTumor-Omics project.

The model uses genomic and epigenomic features to classify samples into one of four existing molecular subtype groups.

## 2. Required Input Files

The prediction script requires two CSV files:

| Input file | Required features |
|---|---:|
| Genomics CSV | 43 genomic features |
| Epigenomics CSV | 20,113 epigenomic features |

Both files must contain the same samples in the same order.

### 2.1 Genomics CSV

- Each row represents one sample.
- Each column represents a genomic feature.
- The input must contain all 43 genomic features used during model training.
- Values must be numeric and encoded as binary values (0 or 1), consistent with the training dataset.
- The first column must contain sample IDs.

### 2.2 Epigenomics CSV

- Each row represents one sample.
- Each column represents an epigenomic feature.
- The input must contain all 20,113 epigenomic features used during model training.
- Values must be numeric.
- The values must follow the same Firehose preprocessing method and numerical scale used in the training dataset.
- Negative values are valid in the training dataset and must not be automatically converted to beta values between 0 and 1.
- The first column must contain sample IDs.

## 3. Sample ID Requirements

- Both files must contain identical sample IDs.
- Sample IDs must appear in the same order in both files.
- Sample IDs must be unique.
- The sample ID should be stored in the first column and used as the CSV index when reading the file.

## 4. Feature Requirements

The model expects the exact feature names and definitions used during training.

The prediction script automatically adds:
- `GEN_` prefix to genomic features.
- `METH_` prefix to epigenomic features.

Therefore, the uploaded CSV column names should contain the original feature names without these prefixes.

All expected features must be present. Additional or missing features may cause prediction to fail.

## 5. Preprocessing Requirements

The saved machine learning pipeline performs the following operations internally:

1. Median imputation for missing values.
2. Removal of zero-variance features.
3. Selection of the top 100 features.
4. Standard scaling.
5. Logistic Regression classification.

Do not independently perform these operations before uploading the files.

The input data must already be processed into the same feature definitions and numerical scale as the training datasets.

## 6. Unsupported Raw Input Formats

The current prediction script does not directly process:

- Raw FASTQ files.
- Raw VCF files.
- BAM files.
- Raw methylation IDAT files.
- Unprocessed 450K or 850K methylation array data.

These