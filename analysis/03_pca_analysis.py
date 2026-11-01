
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Input datasets
DATASETS = {
    "Genomics": "Genomics_Subtype_Aligned.csv",
    "Epigenomics": "Epigenomics_Subtype_Aligned.csv",
    "Transcriptomics": "Transcriptomics_Subtype_Aligned.csv"
}

LABEL_FILE = "Clinical_Subtype_Labels.csv"

# Load subtype labels
labels = pd.read_csv(os.path.join(BASE_DIR, LABEL_FILE))

labels["Patient_ID"] = (
    labels["Sample ID"]
    .astype(str)
    .str.replace(".", "-", regex=False)
    .str[:12]
)

labels = labels.set_index("Patient_ID")

# PCA for each omics dataset
for name, filename in DATASETS.items():

    print(f"\nProcessing {name}...")

    file_path = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(file_path, index_col=0)

    # Align subtype labels with sample order
    y = labels.loc[df.index, "mRNA Subtype Clusters"]

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    variance = pca.explained_variance_ratio_

    pca_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2"],
        index=df.index
    )

    pca_df["Subtype"] = y.values

    print(f"PC1 variance: {variance[0] * 100:.2f}%")
    print(f"PC2 variance: {variance[1] * 100:.2f}%")
    print(f"Total variance: {variance.sum() * 100:.2f}%")

    # Create visualization
    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Subtype",
        s=75,
        alpha=0.85
    )

    plt.title(
        f"{name}: PCA of PCPG Molecular Subtypes",
        fontsize=14
    )

    plt.xlabel(f"PC1 ({variance[0] * 100:.2f}% variance)")
    plt.ylabel(f"PC2 ({variance[1] * 100:.2f}% variance)")

    plt.legend(
        title="Molecular Subtype",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        f"{name}_PCA.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll PCA analyses completed.")
