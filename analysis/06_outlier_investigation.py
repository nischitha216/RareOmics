
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Project directories
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load correlation summary
# --------------------------------------------------

correlation_file = os.path.join(
    TABLES_DIR,
    "Epigenomics_Sample_Correlation_Summary.csv"
)

correlation_df = pd.read_csv(correlation_file)

# --------------------------------------------------
# 3. Load subtype labels
# --------------------------------------------------

labels_file = os.path.join(
    BASE_DIR,
    "Clinical_Subtype_Labels.csv"
)

labels_df = pd.read_csv(labels_file)

labels_df["Patient_ID"] = (
    labels_df["Sample ID"]
    .astype(str)
    .str.replace(".", "-", regex=False)
    .str[:12]
)

# --------------------------------------------------
# 4. Merge correlation and subtype information
# --------------------------------------------------

merged_df = correlation_df.merge(
    labels_df[["Patient_ID", "mRNA Subtype Clusters"]],
    on="Patient_ID",
    how="left"
)

# --------------------------------------------------
# 5. Select 10 lowest-correlating samples
# --------------------------------------------------

lowest_samples = merged_df.nsmallest(
    10,
    "Mean_Correlation"
)

print("\n10 Lowest-Correlating Samples and Their Subtypes:")

print(
    lowest_samples[
        ["Patient_ID", "Mean_Correlation", "mRNA Subtype Clusters"]
    ].to_string(index=False)
)

# Save report
lowest_samples.to_csv(
    os.path.join(
        TABLES_DIR,
        "Epigenomics_Low_Correlation_Samples.csv"
    ),
    index=False
)

# --------------------------------------------------
# 6. Count subtypes among the 10 samples
# --------------------------------------------------

subtype_counts = lowest_samples[
    "mRNA Subtype Clusters"
].value_counts()

print("\nSubtype distribution among the 10 samples:")
print(subtype_counts)

# --------------------------------------------------
# 7. Visualize average correlation by subtype
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=merged_df,
    x="mRNA Subtype Clusters",
    y="Mean_Correlation"
)

plt.title("Epigenomics Sample Correlation by Molecular Subtype")
plt.xlabel("Molecular Subtype")
plt.ylabel("Mean Sample Correlation")

plt.xticks(rotation=20, ha="right")
plt.tight_layout()

output_path = os.path.join(
    FIGURES_DIR,
    "Epigenomics_Correlation_By_Subtype.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nBoxplot saved:", output_path)

print("\nOutlier investigation completed.")
