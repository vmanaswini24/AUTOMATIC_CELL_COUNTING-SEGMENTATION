
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

# Normalize the image to 8-bit intensity
gray = cv2.normalize(
    gray, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Smooth small variations
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Create a binary image using adaptive thresholding
binary = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51,
    3
)

# Create a small structuring element
kernel = np.ones((3, 3), np.uint8)

# Opening removes small foreground noise
opening = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel, iterations=1
)

# Closing fills small gaps in foreground regions
closing = cv2.morphologyEx(
    opening, cv2.MORPH_CLOSE, kernel, iterations=1
)

# Display each stage
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Grayscale Image")

axes[1].imshow(binary, cmap="gray")
axes[1].set_title("Thresholded Image")

axes[2].imshow(opening, cmap="gray")
axes[2].set_title("After Opening")

axes[3].imshow(closing, cmap="gray")
axes[3].set_title("After Closing")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Morphological operations completed.")
