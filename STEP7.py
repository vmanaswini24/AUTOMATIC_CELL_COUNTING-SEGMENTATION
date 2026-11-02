
from pathlib import Path
import cv2
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Locate the dataset
dataset = Path("mCellSeg")
image_folder = dataset / "labeled" / "images"
mask_folder = dataset / "labeled" / "masks"

# Step 2: Select one image and its matching mask
image_path = sorted(image_folder.glob("*.tif"))[0]
mask_path = mask_folder / f"{image_path.stem}_mask.tif"

if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

image = tifffile.imread(str(image_path))
ground_truth = tifffile.imread(str(mask_path))

# Step 3: Convert the image to grayscale
if image.ndim == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
else:
    gray = image.copy()

gray = cv2.normalize(
    gray, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Step 4: Apply thresholding and morphological opening
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

binary = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51, 3
)

kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel
)

# Step 5: Prepare watershed markers
distance = cv2.distanceTransform(
    cleaned, cv2.DIST_L2, 5
)

if distance.max() == 0:
    raise ValueError("No foreground regions detected.")

_, foreground = cv2.threshold(
    distance, 0.35 * distance.max(),
    255, cv2.THRESH_BINARY
)
foreground = foreground.astype(np.uint8)

background = cv2.dilate(
    cleaned, kernel, iterations=3
)
unknown = cv2.subtract(background, foreground)

_, markers = cv2.connectedComponents(foreground)
markers = markers + 1
markers[unknown == 255] = 0

# Step 6: Apply watershed
watershed_input = cv2.cvtColor(
    gray, cv2.COLOR_GRAY2BGR
)

markers = cv2.watershed(
    watershed_input, markers.astype(np.int32)
)

# Step 7: Measure each detected region
# Positive labels greater than 1 are candidate regions.
labels, counts = np.unique(
    markers[markers > 1],
    return_counts=True
)

# Exclude tiny regions that may be noise.
min_area = 20
valid = counts >= min_area

labels = labels[valid]
areas = counts[valid]

# Step 8: Calculate measurements
cell_count = len(labels)
average_area = float(np.mean(areas)) if cell_count else 0.0

print("Image:", image_path.name)
print("Detected cell regions:", cell_count)
print("Average detected area:", round(average_area, 2), "pixels")

# Step 9: Save each region's area
results = pd.DataFrame({
    "Region_ID": labels,
    "Area_pixels": areas
})

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

results.to_csv(
    output_folder / "cell_areas.csv", index=False
)

print("\nFirst 10 detected regions:")
print(results.head(10))
print("\nArea results saved to results/cell_areas.csv")

# Step 10: Display detected boundaries
display_image = cv2.cvtColor(
    gray, cv2.COLOR_GRAY2RGB
)
display_image[markers == -1] = [255, 0, 0]

plt.figure(figsize=(8, 6))
plt.imshow(display_image)
plt.title(f"Watershed Boundaries - Regions: {cell_count}")
plt.axis("off")
plt.show()

# Step 11: Count reference instances from the mask
reference_count = len(np.unique(ground_truth)) - 1

print("\nReference cell count:", reference_count)
print("Predicted region count:", cell_count)
print("Count difference:", abs(reference_count - cell_count))
