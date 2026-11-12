"""
SAM (Segment Anything Model) - Accurate Automatic Image Segmentation
=======================================================================
Uses Meta AI's pretrained Segment Anything Model to automatically find
and segment every distinct object/region in an image (eyes, hair, face,
trees, background, etc.) with high accuracy. No segment ID labeling --
just accurate colored segmentation as requested.

ONE-TIME SETUP (run these in PowerShell before using the script):

    py -m pip install torch torchvision
    py -m pip install git+https://github.com/facebookresearch/segment-anything.git

Then download the model checkpoint (~375 MB, one-time download) by opening
this link in your browser and saving it into the SAME folder as this script:

    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

(This is the "vit_b" model -- smallest/fastest version, works fine on CPU,
 no GPU required. Save the .pth file as-is, don't rename it.)

USAGE:
    py segment_sam.py --input images/Human.jpg --outdir results results .

Output:
    <input>_sam_segmented.png  -> accurately segmented, colored result
"""

import cv2
import numpy as np
import argparse
import os
import sys

CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"


def check_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Model file not found: {checkpoint_path}")
        print("Download it first from:")
        print("  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print(f"and place it in the same folder as this script (as '{CHECKPOINT_NAME}').\n")
        sys.exit(1)


def segment_with_sam(image, checkpoint_path):
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM model on device: {device} (this can take 10-30 seconds)...")

    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=300,    # ignore tiny noise regions
        crop_n_layers=1,             # also scans at a second zoom level -> catches whole-face/whole-shirt regions better
    )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Running segmentation (this can take 1-3 minutes on CPU)...")
    masks = mask_generator.generate(image_rgb)
    print(f"Found {len(masks)} raw candidate regions.")
    return masks


def draw_sam_segments(image, masks, alpha=0.6):
    h, w = image.shape[:2]
    overlay = image.copy()
    painted = np.zeros((h, w), dtype=bool)

    # Largest regions first, so coherent big areas (face, shirt, background)
    # get painted as solid blocks BEFORE any smaller overlapping fragments can
    # cut into them. Each pixel is only ever colored once.
    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)

    rng = np.random.default_rng(42)
    used_regions = 0
    for m in masks_sorted:
        seg_mask = m["segmentation"]
        new_pixels = seg_mask & (~painted)
        if new_pixels.sum() < 150:   # skip if this mask adds almost nothing new
            continue
        color = rng.integers(40, 255, size=3).tolist()
        overlay[new_pixels] = color
        painted |= new_pixels
        used_regions += 1

    # Fill any leftover uncovered pixels (e.g. face if SAM missed it) with one
    # neutral "uncategorized" color instead of leaving the original photo showing.
    leftover = ~painted
    if leftover.sum() > 0:
        overlay[leftover] = (120, 120, 120)  # neutral gray = "uncategorized region"

    print(f"Painted {used_regions} solid non-overlapping segments "
          f"({leftover.sum()} px left as uncategorized).")

    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return output


def main():
    parser = argparse.ArgumentParser(description="Accurate SAM-based image segmentation")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default=CHECKPOINT_NAME, help="Path to SAM checkpoint .pth file")
    parser.add_argument("--outdir results", default=".")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    check_checkpoint(args.checkpoint)
    masks = segment_with_sam(image, args.checkpoint)
    result = draw_sam_segments(image, masks)

    base = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{base}_sam_segmented.png")
    cv2.imwrite(out_path, result)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
