
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Set project directories
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load epigenomics dataset
# --------------------------------------------------

file_path = os.path.join(
    BASE_DIR,
    "Epigenomics_Subtype_Aligned.csv"
)

df = pd.read_csv(file_path, index_col=0)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. Handle missing values
# --------------------------------------------------

# Replace missing values with the median of each gene
df = df.apply(
    lambda column: column.fillna(column.median()),
    axis=0
)

print("Missing values after imputation:", df.isna().sum().sum())

# --------------------------------------------------
# 4. Calculate sample-to-sample correlation
# --------------------------------------------------

# Rows represent genes and columns represent samples.
# Transpose so that rows represent samples.

correlation_matrix = df.T.corr(method="pearson")

# Save correlation matrix
correlation_path = os.path.join(
    TABLES_DIR,
    "Epigenomics_Sample_Correlation.csv"
)

correlation_matrix.to_csv(correlation_path)

print("Correlation matrix saved:", correlation_path)

# --------------------------------------------------
# 5. Generate correlation heatmap
# --------------------------------------------------

plt.figure(figsize=(13, 11))

sns.heatmap(
    correlation_matrix,
    cmap="vlag",
    center=0,
    xticklabels=False,
    yticklabels=False
)

plt.title("Epigenomics: Sample-to-Sample Correlation")
plt.xlabel("Samples")
plt.ylabel("Samples")

plt.tight_layout()

heatmap_path = os.path.join(
    FIGURES_DIR,
    "Epigenomics_Correlation_Heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Heatmap saved:", heatmap_path)

# --------------------------------------------------
# 6. Calculate average correlation per sample
# --------------------------------------------------

# Create a separate writable NumPy array.
# This avoids modifying the original correlation matrix.

correlation_values = correlation_matrix.to_numpy(copy=True)

# Exclude self-correlation (diagonal values)
np.fill_diagonal(correlation_values, np.nan)

# Calculate average correlation with all other samples
mean_correlations = np.nanmean(
    correlation_values,
    axis=1
)

# Create summary table
outlier_report = pd.DataFrame({
    "Patient_ID": correlation_matrix.index,
    "Mean_Correlation": mean_correlations
})

# Sort from lowest to highest correlation
outlier_report = outlier_report.sort_values(
    by="Mean_Correlation",
    ascending=True
)

# --------------------------------------------------
# 7. Save sample correlation summary
# --------------------------------------------------

summary_path = os.path.join(
    TABLES_DIR,
    "Epigenomics_Sample_Correlation_Summary.csv"
)

outlier_report.to_csv(
    summary_path,
    index=False
)

print("\nSample correlation summary saved:", summary_path)

# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\nLowest 10 average sample correlations:")

print(
    outlier_report.head(10).to_string(index=False)
)

print("\nHighest 10 average sample correlations:")

print(
    outlier_report.tail(10).sort_values(
        by="Mean_Correlation",
        ascending=False
    ).to_string(index=False)
)

print(
    "\nOverall mean sample correlation:",
    round(np.nanmean(mean_correlations), 6)
)

print("\nEpigenomics correlation analysis completed.")
