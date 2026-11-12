"""
Region Swap - Replace ONE segment's content with a different image/texture
=============================================================================
Takes your original image + a mask marking one specific region, and fills
that region with content from a replacement image/texture instead -- with
feathered edges so the swap blends naturally instead of looking pasted.

TWO WAYS TO GET THE MASK FOR THE REGION YOU WANT TO SWAP:

  MODE 1 - "sam": click one point inside the object (needs SAM already set up)
      py region_swap.py --input Human.jpg --replacement images/clouds_sky.png \
          --mode sam --point 300 600 --outdir .

  MODE 2 - "label": reuse a label-map mask you already generated earlier
      (e.g. shapes_test_mask_felzenszwalb.png) + the specific ID you want
      py region_swap.py --input shapes_test.png --replacement images/clouds_sky.png \
          --mode label --mask_file shapes_test_mask_felzenszwalb.png --mask_id 3 --outdir .

Output:
    <input>_swapped.png -> original image with that one region replaced
"""

import cv2
import numpy as np
import argparse
import os
import sys

MODEL_TYPE = "vit_b"
DEFAULT_CHECKPOINT = "sam_vit_b_01ec64.pth"


def get_mask_sam(image, points, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: SAM checkpoint not found: {checkpoint_path}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n")
        sys.exit(1)

    import torch
    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM on {device}...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Encoding image...")
    predictor.set_image(image_rgb)

    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    for (x, y) in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)]
        combined_mask |= best_mask
        print(f"  point ({x},{y}) added {best_mask.sum()} px")

    return combined_mask.astype(np.uint8) * 255


def get_mask_label(label_mask_path, target_id):
    label_map = cv2.imread(label_mask_path, cv2.IMREAD_GRAYSCALE)
    if label_map is None:
        raise FileNotFoundError(f"Could not read mask file: {label_mask_path}")
    # NOTE: earlier scripts saved label maps NORMALIZED to 0-255 for viewing,
    # so exact original IDs may not match 1:1. This mode works best with a
    # raw/unnormalized label mask. If your swap looks wrong, use --mode sam instead.
    mask = (label_map == target_id).astype(np.uint8) * 255
    return mask


def feather_mask(mask, blur_px=15):
    mask_f = mask.astype(np.float32) / 255.0
    feathered = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=blur_px)
    return np.clip(feathered, 0, 1)


def swap_region(image, mask, replacement):
    h, w = image.shape[:2]
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        raise ValueError("Mask is empty -- no region found to swap. Check your point/ID.")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    box_w, box_h = x1 - x0 + 1, y1 - y0 + 1

    # Resize replacement texture/image to cover the bounding box of the region
    repl_resized = cv2.resize(replacement, (box_w, box_h), interpolation=cv2.INTER_LINEAR)

    # Build full-size canvas with replacement placed at the right location
    repl_canvas = np.zeros_like(image)
    repl_canvas[y0:y1 + 1, x0:x1 + 1] = repl_resized

    # Feather mask edges for a natural blend instead of a hard paste
    alpha = feather_mask(mask, blur_px=8)
    alpha_3ch = np.dstack([alpha] * 3)

    result = (repl_canvas.astype(np.float32) * alpha_3ch +
              image.astype(np.float32) * (1 - alpha_3ch))
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Swap one segment's content with another image/texture")
    parser.add_argument("--input", required=True)
    parser.add_argument("--replacement", required=True, help="Image/texture to fill the region with")
    parser.add_argument("--mode", choices=["sam", "label"], required=True)
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Y"), action="append",
                         help="Point inside the region (repeat --point multiple times to merge several parts into one mask)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mask_file", help="Required for --mode label")
    parser.add_argument("--mask_id", type=int, help="Required for --mode label")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    replacement = cv2.imread(args.replacement)
    if replacement is None:
        raise FileNotFoundError(f"Could not read replacement image: {args.replacement}")

    if args.mode == "sam":
        if not args.point:
            sys.exit("At least one --point X Y is required for --mode sam")
        mask = get_mask_sam(image, args.point, args.checkpoint)
    else:
        if not args.mask_file or args.mask_id is None:
            sys.exit("--mask_file and --mask_id are required for --mode label")
        mask = get_mask_label(args.mask_file, args.mask_id)

    result = swap_region(image, mask, replacement)

    base = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{base}_swapped.png")
    cv2.imwrite(out_path, result)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
