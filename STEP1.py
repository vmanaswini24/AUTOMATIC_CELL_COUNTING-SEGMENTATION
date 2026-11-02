
from pathlib import Path

# Find the dataset folder inside your project
dataset = Path("mCellSeg")

# Define the image and mask folders
image_folder = dataset / "labeled" / "images"
mask_folder = dataset / "labeled" / "masks"

# Find all TIFF images
images = sorted(image_folder.glob("*.tif"))
masks = sorted(mask_folder.glob("*.tif"))

# Display the dataset information
print("Images found:", len(images))
print("Masks found:", len(masks))

# Display a few filenames
print("\nSample images:")
for file in images[:5]:
    print(file.name)

print("\nSample masks:")
for file in masks[:5]:
    print(file.name)
