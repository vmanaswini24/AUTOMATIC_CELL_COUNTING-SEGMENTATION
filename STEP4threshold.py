
from pathlib import Path
import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# Load one microscopy image
dataset = Path("mCellSeg")
image_path = sorted(
    (dataset / "labeled" / "images").glob("*.tif")
)[0]

image = tifffile.imread(str(image_path))

# Convert to grayscale if needed
if image.ndim == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
else:
    gray = image.copy()

# Normalize image intensity
gray = cv2.normalize(
    gray, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Otsu thresholding
_, otsu = cv2.threshold(
    blurred, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Adaptive thresholding
adaptive = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51,
    3
)

# Display the comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Grayscale Image")

axes[1].imshow(otsu, cmap="gray")
axes[1].set_title("Otsu Threshold")

axes[2].imshow(adaptive, cmap="gray")
axes[2].set_title("Adaptive Threshold")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Thresholding comparison completed.")
