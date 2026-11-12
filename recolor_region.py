"""
Recolor Region - Change the COLOR of one segment while keeping its
shading, folds, and texture natural (e.g. recolor a shirt from blue to red,
but wrinkles/shadows/highlights stay exactly where they are).

HOW IT WORKS:
Converts the image to HSV. Hue+Saturation control what "color" something
is; Value (brightness) controls shading/folds/lighting. This script only
replaces Hue+Saturation inside your masked region and leaves Value
untouched -- so all the natural shading stays, just the color changes.

USAGE:
  py recolor_region.py --input images/Human.jpg  --mode sam --point 91 330 --color red --outdir results results results .
  py recolor_region.py --input images/Human.jpg  --mode sam --point 91 330 --color 40,120,200 --outdir results results results results .
      (--color accepts a name OR a specific B,G,R value)

  py recolor_region.py --input images/shapes_test.png --mode label \
      --mask_file shapes_raw_labels.png --mask_id 3 --color green --outdir results results results results .

Output:
    <input>_recolored.png
"""

import cv2
import numpy as np
import argparse
import os
import sys

MODEL_TYPE = "vit_b"
DEFAULT_CHECKPOINT = "sam_vit_b_01ec64.pth"

NAMED_COLORS_BGR = {
    "red": (40, 40, 220),
    "blue": (220, 100, 30),
    "green": (40, 180, 40),
    "yellow": (30, 220, 230),
    "orange": (20, 130, 240),
    "purple": (200, 40, 160),
    "pink": (180, 120, 240),
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "gray": (130, 130, 130),
    "brown": (30, 70, 110),
}


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
    mask = (label_map == target_id).astype(np.uint8) * 255
    return mask


def parse_color(color_arg):
    if color_arg in NAMED_COLORS_BGR:
        return NAMED_COLORS_BGR[color_arg]
    try:
        b, g, r = [int(v) for v in color_arg.split(",")]
        return (b, g, r)
    except Exception:
        sys.exit(f"Invalid --color '{color_arg}'. Use a name ({list(NAMED_COLORS_BGR)}) or 'B,G,R' like 40,120,200")


def recolor_region(image, mask, target_bgr, feather_px=6):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)

    recolored_hsv = hsv.copy()
    recolored_hsv[..., 0] = target_hsv[0]   # Hue
    recolored_hsv[..., 1] = target_hsv[1]   # Saturation
    # Value (channel 2) untouched -> shading/folds/highlights preserved

    recolored_bgr = cv2.cvtColor(recolored_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Feather mask edges for smooth blend
    mask_f = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=feather_px)
    alpha = np.clip(alpha, 0, 1)
    alpha_3ch = np.dstack([alpha] * 3)

    result = (recolored_bgr.astype(np.float32) * alpha_3ch +
              image.astype(np.float32) * (1 - alpha_3ch))
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Recolor one segment while preserving its shading/texture")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", choices=["sam", "label"], required=True)
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Y"), action="append",
                         help="Point inside the region (repeat to merge multiple parts)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mask_file")
    parser.add_argument("--mask_id", type=int)
    parser.add_argument("--color", required=True, help="Color name or 'B,G,R' value")
    parser.add_argument("--outdir results results results", default=".")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    if args.mode == "sam":
        if not args.point:
            sys.exit("At least one --point X Y is required for --mode sam")
        mask = get_mask_sam(image, args.point, args.checkpoint)
    else:
        if not args.mask_file or args.mask_id is None:
            sys.exit("--mask_file and --mask_id are required for --mode label")
        mask = get_mask_label(args.mask_file, args.mask_id)

    target_bgr = parse_color(args.color)
    result = recolor_region(image, mask, target_bgr)

    base = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{base}_recolored.png")
    cv2.imwrite(out_path, result)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
