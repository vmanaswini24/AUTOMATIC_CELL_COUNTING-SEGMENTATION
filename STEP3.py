
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

# Handle grayscale, RGB, and RGBA images
if image.ndim == 2:
    gray = image.copy()
elif image.shape[-1] == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
elif image.shape[-1] == 4:
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
else:
    raise ValueError(f"Unsupported image shape: {image.shape}")

# Normalize intensity values for display and processing
gray = cv2.normalize(
    gray, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Reduce small-scale noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Display preprocessing results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")

axes[1].imshow(gray, cmap="gray")
axes[1].set_title("Grayscale Image")

axes[2].imshow(blurred, cmap="gray")
axes[2].set_title("Gaussian Blurred Image")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Preprocessing completed.")
print("Image dimensions:", gray.shape)
