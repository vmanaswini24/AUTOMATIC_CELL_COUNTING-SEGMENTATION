
from pathlib import Path
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the dataset location
dataset = Path("mCellSeg")

image_folder = dataset / "labeled" / "images"
mask_folder = dataset / "labeled" / "masks"

# Step 2: Find the TIFF images and masks
images = sorted(image_folder.glob("*.tif"))
masks = sorted(mask_folder.glob("*.tif"))

print("Images found:", len(images))
print("Masks found:", len(masks))

# Step 3: Check that files are available
if len(images) == 0 or len(masks) == 0:
    print("No files found!")
    print("Please check the dataset folder path.")
else:
    # Step 4: Select the first image
    image_path = images[0]

    # Step 5: Find the corresponding mask
    matching_masks = [
    mask for mask in masks
    if mask.stem == image_path.stem + "_mask"
]

    if not matching_masks:
        print("No matching mask found for:", image_path.name)
        print("Image filename:", image_path.name)
        print("Sample mask filenames:",
              [m.name for m in masks[:5]])
    else:
        mask_path = matching_masks[0]

        # Step 6: Load the image and mask
        image = tifffile.imread(str(image_path))
        ground_truth = tifffile.imread(str(mask_path))

        print("Image:", image_path.name)
        print("Image shape:", image.shape)
        print("Mask shape:", ground_truth.shape)
        print("Unique mask labels:",
              len(np.unique(ground_truth)) - 1)

        # Step 7: Display the original image and mask
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original Microscopic Image")
        axes[0].axis("off")

        axes[1].imshow(ground_truth, cmap="nipy_spectral")
        axes[1].set_title("Ground-Truth Cell Mask")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
