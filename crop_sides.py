"""Crop left and right background around a single centered wine bottle.

Approach:
- Use Canny edge detection and vertical morphological closing to highlight bottle edges.
- Compute per-column edge counts and find the largest contiguous column segment with strong edge activity.
- Crop image to that segment (with small margins) to remove left/right background (white or wood).

Usage (PowerShell):
    python crop_sides.py --in .\images --out .\images\cropped_sides --debug
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _auto_canny_thresholds(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower == upper:
        lower = max(0, int(0.5 * v))
        upper = min(255, int(1.5 * v))
    return lower, upper


def detect_vertical_bounds(img, resize_w=1000, debug=False):
    """Return (left, right, debug_image) column bounds for bottle region.

    If detection fails, returns (0, width, debug_image).
    """
    h0, w0 = img.shape[:2]
    scale = 1.0
    if w0 > resize_w:
        scale = resize_w / float(w0)
        img_r = cv2.resize(img, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
    else:
        img_r = img.copy()

    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    lower, upper = _auto_canny_thresholds(blur)
    edges = cv2.Canny(blur, lower, upper)

    # Close vertically to connect edge fragments of bottle sides
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(3, int(0.02 * img_r.shape[0]))))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    col_counts = (edges_closed > 0).sum(axis=0)

    h, w = img_r.shape[:2]
    # thresholds: at least a few edge pixels or a small fraction of the max
    min_col = max(3, int(0.008 * h))
    if col_counts.max() > 0:
        thresh = max(min_col, int(0.05 * col_counts.max()))
    else:
        thresh = min_col

    cols = np.where(col_counts >= thresh)[0]

    # if no strong edge columns, attempt alternate mask-based detection
    used_edges = True
    if cols.size == 0:
        used_edges = False
        # Otsu-based foreground mask (bottle darker / textured than background)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = 255 - th
        col_counts2 = (mask > 0).sum(axis=0)
        cols = np.where(col_counts2 > max(1, int(0.01 * h)))[0]

    if cols.size == 0:
        # detection failed; return full image
        return 0, w0, (img, edges, edges_closed)

    # find contiguous segments among columns
    segments = []
    start = cols[0]
    prev = cols[0]
    for c in cols[1:]:
        if c == prev + 1:
            prev = c
        else:
            segments.append((start, prev))
            start = c
            prev = c
    segments.append((start, prev))

    # choose the largest segment (by width)
    best = max(segments, key=lambda s: s[1] - s[0])
    left_r, right_r = best[0], best[1] + 1

    # expand a small margin to avoid clipping the label
    margin = int(0.03 * w)
    left_r = max(0, left_r - margin)
    right_r = min(w, right_r + margin)

    # map back to original coordinates
    inv_scale = 1.0 / scale
    left0 = int(left_r * inv_scale)
    right0 = int(right_r * inv_scale)
    left0 = max(0, min(left0, w0 - 1))
    right0 = max(1, min(right0, w0))

    dbg_img = None
    if debug:
        # overlay edges and vertical lines on a copy of the resized image
        overlay = img_r.copy()
        # edges in red
        overlay[edges > 0] = (0, 0, 255)
        # draw chosen vertical bounds scaled to resized coords
        cv2.line(overlay, (left_r, 0), (left_r, h), (0, 255, 0), 2)
        cv2.line(overlay, (right_r, 0), (right_r, h), (0, 255, 0), 2)
        dbg_img = overlay

    return left0, right0, (dbg_img, edges, edges_closed, used_edges)


def get_bottle_bbox(img, resize_w=1000):
    """Return bounding box (left, right) in original image coords for the largest foreground contour (bottle).

    Returns None if no contour found.
    """
    h0, w0 = img.shape[:2]
    scale = 1.0
    if w0 > resize_w:
        scale = resize_w / float(w0)
        img_r = cv2.resize(img, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
    else:
        img_r = img.copy()

    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = 255 - th
    # close holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    inv_scale = 1.0 / scale
    x0 = int(x * inv_scale)
    x1 = int((x + w) * inv_scale)
    x0 = max(0, min(x0, w0 - 1))
    x1 = max(1, min(x1, w0))
    return x0, x1


def process_image(path, out_dir, resize_w=1000, debug=False):
    img = cv2.imread(str(path))
    if img is None:
        print(f"Skipped (can't read): {path}")
        return

    left, right, dbg = detect_vertical_bounds(img, resize_w=resize_w, debug=debug)
    h, w = img.shape[:2]
    # ensure bottle is fully inside crop: expand bounds to include bottle mask bbox if needed
    bbox = get_bottle_bbox(img, resize_w=resize_w)
    if bbox is not None:
        bx_left, bx_right = bbox
        margin_px = max(2, int(0.02 * w))
        desired_left = max(0, bx_left - margin_px)
        desired_right = min(w, bx_right + margin_px)
        # if detected bounds cut into the bottle, expand to include full bottle bbox
        if bx_left < left:
            left = min(left, desired_left)
            left = max(0, desired_left)
        if bx_right > right:
            right = max(right, desired_right)
    # safety: if detection yields full width, try small inward margins
    if left == 0 and right == w:
        # try small center crop to avoid leaving same image
        margin = int(0.02 * w)
        left = margin
        right = w - margin

    cropped = img[:, left:right]

    ensure_dir(out_dir)
    # preserve original filename (keep extension)
    out_path = Path(out_dir) / Path(path).name
    cv2.imwrite(str(out_path), cropped)

    if debug:
        dbg_dir = Path(out_dir) / "debug"
        ensure_dir(dbg_dir)
        dbg_img = dbg[0]
        if dbg_img is None:
            # create a simple visualization from edges
            edges = dbg[1] if len(dbg) > 1 else None
            if edges is not None:
                viz = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            else:
                viz = img.copy()
        else:
            viz = dbg_img
        cv2.imwrite(str(dbg_dir / (Path(path).stem + "_debug_resized.jpg")), viz)
        # also save a crop preview
        cv2.imwrite(str(dbg_dir / (Path(path).stem + "_cropped.jpg")), cropped)

    print(f"Saved: {out_path}")


def process_directory(in_dir, out_dir, resize_w=1000, debug=False):
    p = Path(in_dir)
    ensure_dir(out_dir)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    files = []
    for e in exts:
        files.extend(sorted(p.glob(e)))

    files = [f for f in files if f.is_file() and Path(out_dir) not in f.parents]

    if not files:
        print(f"No images found in {in_dir}")
        return

    for f in files:
        process_image(f, out_dir, resize_w=resize_w, debug=debug)


def main():
    parser = argparse.ArgumentParser(description="Crop left/right backgrounds around centered wine bottles")
    parser.add_argument("--in", dest="indir", default="images", help="Input images directory")
    parser.add_argument("--out", dest="outdir", default=os.path.join("images", "cropped_sides"), help="Output directory")
    parser.add_argument("--resize", dest="resize_w", type=int, default=1000, help="Resize width for detection (speed/accuracy tradeoff)")
    parser.add_argument("--debug", action="store_true", help="Save debug visualizations")
    args = parser.parse_args()
    process_directory(args.indir, args.outdir, resize_w=args.resize_w, debug=args.debug)


if __name__ == "__main__":
    main()
