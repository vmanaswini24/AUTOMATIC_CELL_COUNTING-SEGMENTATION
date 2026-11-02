
from pathlib import Path
import cv2
import tifffile
import numpy as np
import pandas as pd

# Dataset paths
dataset = Path("mCellSeg")
image_folder = dataset / "labeled" / "images"
mask_folder = dataset / "labeled" / "masks"

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

image_paths = sorted(image_folder.glob("*.tif"))

if not image_paths:
    raise FileNotFoundError(
        "No TIFF images found. Check your dataset path."
    )

records = []

# Process each image
for index, image_path in enumerate(image_paths, start=1):
    mask_path = mask_folder / f"{image_path.stem}_mask.tif"

    if not mask_path.exists():
        print("Skipping image without mask:", image_path.name)
        continue

    image = tifffile.imread(str(image_path))
    ground_truth = tifffile.imread(str(mask_path))

    # Convert to grayscale
    if image.ndim == 3:
        if image.shape[-1] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[-1] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            print("Skipping unsupported image:", image_path.name)
            continue
    elif image.ndim == 2:
        gray = image.copy()
    else:
        print("Skipping unsupported image:", image_path.name)
        continue

    # Check mask compatibility
    if ground_truth.ndim != 2 or gray.shape != ground_truth.shape:
        print("Skipping incompatible mask:", mask_path.name)
        continue

    # Normalize and preprocess
    gray = cv2.normalize(
        gray, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 3
    )

    # Morphological opening
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel
    )

    # Connected-component count and area measurements
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(cleaned, 8)
    )

    min_area = 20
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_areas = areas[areas >= min_area]

    predicted_count = len(valid_areas)
    average_area = (
        float(np.mean(valid_areas))
        if predicted_count else 0.0
    )

    # Pixel-level segmentation evaluation
    pred = cleaned > 0
    truth = ground_truth > 0

    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator else 1.0
    )

    iou = (
        tp / iou_denominator
        if iou_denominator else 1.0
    )

    reference_count = len(np.unique(ground_truth)) - 1

    records.append({
        "Image": image_path.name,
        "Predicted_Regions": predicted_count,
        "Reference_Instances": reference_count,
        "Average_Area_Pixels": round(average_area, 2),
        "Dice_Score": round(float(dice), 4),
        "IoU": round(float(iou), 4)
    })

    print(f"Processed {index}/{len(image_paths)}: {image_path.name}")

# Save results
df = pd.DataFrame(records)
csv_path = output_folder / "final_results.csv"
df.to_csv(csv_path, index=False)

print("\nBatch processing completed!")
print("Images evaluated:", len(df))
print("Results saved to:", csv_path)

if not df.empty:
    print("\nMean predicted regions:", round(
        df["Predicted_Regions"].mean(), 2
    ))
    print("Mean reference instances:", round(
        df["Reference_Instances"].mean(), 2
    ))
    print("Mean Dice score:", round(
        df["Dice_Score"].mean(), 4
    ))
    print("Mean IoU:", round(df["IoU"].mean(), 4))
