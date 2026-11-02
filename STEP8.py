
from pathlib import Path
import cv2
import tifffile
import numpy as np

# Step 1: Locate the dataset
dataset = Path("mCellSeg")
image_folder = dataset / "labeled" / "images"
mask_folder = dataset / "labeled" / "masks"

# Step 2: Load one image and its matching mask
image_path = sorted(image_folder.glob("*.tif"))[0]
mask_path = mask_folder / f"{image_path.stem}_mask.tif"

if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

image = tifffile.imread(str(image_path))
ground_truth = tifffile.imread(str(mask_path))

# Step 3: Convert image to grayscale
if image.ndim == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
else:
    gray = image.copy()

gray = cv2.normalize(
    gray, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Step 4: Generate a predicted binary segmentation
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

predicted = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51,
    3
)

kernel = np.ones((3, 3), np.uint8)
predicted = cv2.morphologyEx(
    predicted, cv2.MORPH_OPEN, kernel
)

# Step 5: Convert prediction and reference mask
# into binary foreground/background masks.
pred = predicted > 0
truth = ground_truth > 0

if pred.shape != truth.shape:
    raise ValueError(
        f"Shape mismatch: prediction {pred.shape}, "
        f"reference {truth.shape}"
    )

# Step 6: Calculate pixel-level metrics
tp = np.logical_and(pred, truth).sum()
fp = np.logical_and(pred, ~truth).sum()
fn = np.logical_and(~pred, truth).sum()

dice = (2 * tp) / (2 * tp + fp + fn) \
    if (2 * tp + fp + fn) else 1.0

iou = tp / (tp + fp + fn) \
    if (tp + fp + fn) else 1.0

# Step 7: Display results
print("Image:", image_path.name)
print(f"Dice Score: {dice:.4f} ({dice * 100:.2f}%)")
print(f"IoU: {iou:.4f} ({iou * 100:.2f}%)")

# Step 8: Visualize the comparison
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Image")

axes[1].imshow(truth, cmap="gray")
axes[1].set_title("Ground-Truth Mask")

axes[2].imshow(pred, cmap="gray")
axes[2].set_title("Predicted Binary Mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
