
from pathlib import Path
import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load a microscopy image
dataset = Path("mCellSeg")
image_path = sorted(
    (dataset / "labeled" / "images").glob("*.tif")
)[0]

image = tifffile.imread(str(image_path))

# Step 2: Convert to grayscale
if image.ndim == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
else:
    gray = image.copy()

# Normalize image intensity
gray = cv2.normalize(
    gray, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Step 3: Smooth the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Create a binary image
binary = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51,
    3
)

# Step 5: Clean small noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel, iterations=1
)

# Step 6: Calculate the distance transform
distance = cv2.distanceTransform(
    opening, cv2.DIST_L2, 5
)

# Step 7: Find likely cell-center regions
_, sure_foreground = cv2.threshold(
    distance,
    0.35 * distance.max(),
    255,
    cv2.THRESH_BINARY
)

sure_foreground = sure_foreground.astype(np.uint8)

# Step 8: Identify uncertain boundary regions
sure_background = cv2.dilate(
    opening, kernel, iterations=3
)

unknown = cv2.subtract(
    sure_background, sure_foreground
)

# Step 9: Label the foreground markers
_, markers = cv2.connectedComponents(
    sure_foreground
)

# Background marker starts at 1
markers = markers + 1

# Mark uncertain pixels as 0
markers[unknown == 255] = 0

# Step 10: Apply watershed segmentation
image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

markers = cv2.watershed(
    image_bgr, markers.astype(np.int32)
)

# Mark watershed boundaries in red
result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
result[markers == -1] = [255, 0, 0]

# Display the results
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Grayscale")

axes[1].imshow(opening, cmap="gray")
axes[1].set_title("Preprocessed Binary")

axes[2].imshow(distance, cmap="viridis")
axes[2].set_title("Distance Transform")

axes[3].imshow(result)
axes[3].set_title("Watershed Boundaries")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Watershed segmentation completed.")
print("Positive watershed regions:",
      len(np.unique(markers[markers > 1])))
