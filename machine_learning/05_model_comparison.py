
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIGURE_DIR = os.path.join(BASE_DIR, "results", "figures")
TABLE_DIR = os.path.join(BASE_DIR, "results", "tables")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Results from the six completed models
data = [
    ["Genomics", "Logistic Regression", 0.404706, 0.326059, 0.373819],
    ["Genomics", "Random Forest",       0.438824, 0.341119, 0.375888],
    ["Epigenomics", "Logistic Regression", 0.838151, 0.766178, 0.771593],
    ["Epigenomics", "Random Forest",       0.849580, 0.767519, 0.755082],
    ["Integrated", "Logistic Regression",  0.861008, 0.798833, 0.794093],
    ["Integrated", "Random Forest",        0.849412, 0.759935, 0.757857],
]

columns = [
    "Data Type",
    "Algorithm",
    "Accuracy",
    "Macro F1",
    "Balanced Accuracy"
]

df = pd.DataFrame(data, columns=columns)

# Save numerical results
csv_path = os.path.join(
    TABLE_DIR,
    "Final_Model_Comparison.csv"
)

df.to_csv(csv_path, index=False)

# Prepare grouped bar chart
metrics = [
    "Accuracy",
    "Macro F1",
    "Balanced Accuracy"
]

labels = [
    f"{row['Data Type']}\n{row['Algorithm']}"
    for _, row in df.iterrows()
]

x = np.arange(len(df))
width = 0.24

fig, ax = plt.subplots(figsize=(14, 7))

for i, metric in enumerate(metrics):
    bars = ax.bar(
        x + (i - 1) * width,
        df[metric],
        width,
        label=metric
    )

    ax.bar_label(
        bars,
        labels=[f"{v:.2f}" for v in df[metric]],
        padding=3,
        fontsize=8
    )

ax.set_title(
    "Performance Comparison of Genomics, Epigenomics and Integrated Models",
    fontsize=14,
    pad=18
)

ax.set_ylabel("Cross-validation Score")
ax.set_xlabel("Dataset and Algorithm")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)

ax.set_ylim(0, 1.08)
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plot_path = os.path.join(
    FIGURE_DIR,
    "Final_Model_Comparison.png"
)

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nFINAL MODEL COMPARISON")
print(df.to_string(index=False))

print("\nBest model based on mean Macro-F1:")
best = df.loc[df["Macro F1"].idxmax()]
print(best.to_string())

print("\nSaved:")
print(plot_path)
print(csv_path)
