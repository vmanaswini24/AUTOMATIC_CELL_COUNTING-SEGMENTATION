
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load the batch-processing results
csv_path = Path("results") / "final_results.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        "Run step9_batch_results.py first."
    )

df = pd.read_csv(csv_path)

if df.empty:
    raise ValueError("The results CSV contains no data.")

# Display summary statistics
print("Total images evaluated:", len(df))
print("\nSummary statistics:")
print(df[[
    "Predicted_Regions",
    "Reference_Instances",
    "Average_Area_Pixels",
    "Dice_Score",
    "IoU"
]].describe())

# Graph 1: Predicted vs reference counts for first 20 images
sample = df.head(20)
x = range(len(sample))

plt.figure(figsize=(12, 5))
plt.plot(x, sample["Predicted_Regions"], marker="o",
         label="Predicted Regions")
plt.plot(x, sample["Reference_Instances"], marker="s",
         label="Reference Instances")
plt.xlabel("Image Index")
plt.ylabel("Count")
plt.title("Predicted vs Reference Cell Counts")
plt.legend()
plt.tight_layout()
plt.savefig("results/cell_count_comparison.png", dpi=300)
plt.show()

# Graph 2: Dice scores
plt.figure(figsize=(10, 5))
plt.hist(df["Dice_Score"], bins=15)
plt.xlabel("Dice Score")
plt.ylabel("Number of Images")
plt.title("Distribution of Dice Scores")
plt.tight_layout()
plt.savefig("results/dice_score_distribution.png", dpi=300)
plt.show()

# Graph 3: IoU scores
plt.figure(figsize=(10, 5))
plt.hist(df["IoU"], bins=15)
plt.xlabel("IoU")
plt.ylabel("Number of Images")
plt.title("Distribution of IoU Scores")
plt.tight_layout()
plt.savefig("results/iou_distribution.png", dpi=300)
plt.show()

# Graph 4: Average detected region area
plt.figure(figsize=(10, 5))
plt.hist(df["Average_Area_Pixels"], bins=15)
plt.xlabel("Average Region Area (pixels)")
plt.ylabel("Number of Images")
plt.title("Distribution of Average Region Areas")
plt.tight_layout()
plt.savefig("results/average_area_distribution.png", dpi=300)
plt.show()

print("\nGraphs saved in the results folder.")
