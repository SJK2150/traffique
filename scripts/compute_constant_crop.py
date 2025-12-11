"""Compute a constant safe crop box from per-frame transforms or by estimating frame shifts.

Usage:
 python scripts/compute_constant_crop.py --transforms output/stabilize_cache/smoothed_transforms_9861_10236.npy --video D1F1_stab.mp4 --out output/constant_crop.json

If a transforms file is provided, this will compute the max per-axis translations and produce
a crop box that removes any shifting borders so the cropped frame is constant across the window.
If no transforms are provided, it will fall back to estimating vertical shifts using template
matching on the central band of the video.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import cv2


def compute_from_transforms(trans_path, video_path, out_path):
    data = np.load(str(trans_path), allow_pickle=True).item()
    # data is expected to be dict frame->3x3 matrix
    txs = []
    tys = []
    for k in sorted(data.keys()):
        M = np.array(data[k], dtype=np.float32)
        txs.append(float(M[0, 2]))
        tys.append(float(M[1, 2]))

    txs = np.array(txs)
    tys = np.array(tys)

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    min_tx, max_tx = float(txs.min()), float(txs.max())
    min_ty, max_ty = float(tys.min()), float(tys.max())

    # positive tx means shift right; to avoid black borders we remove margins equal to
    # the max positive right shift and max negative (left) shift etc.
    left_crop = int(math.ceil(max(0.0, max_tx)))
    right_crop = int(math.ceil(max(0.0, -min_tx)))
    top_crop = int(math.ceil(max(0.0, max_ty)))
    bottom_crop = int(math.ceil(max(0.0, -min_ty)))

    x1 = left_crop
    y1 = top_crop
    x2 = max(1, w - right_crop)
    y2 = max(1, h - bottom_crop)

    # clamp
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    box = [x1, y1, x2, y2]
    out = {'constant_crop': box, 'source': str(trans_path)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, 'w'))
    print(f'Wrote constant crop: {out_path} -> {box}')
    return box


def estimate_shift_fallback(video_path, out_path, sample_frames=100):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idxs = np.linspace(0, max(0, total - 1), min(sample_frames, max(1, total)), dtype=int)

    # pick a central template band (center 50% height)
    band_h = int(h * 0.5)
    band_y = int((h - band_h) / 2)

    ref_idx = sample_idxs[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ref_idx))
    ret, ref = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError('Cannot read video for fallback')
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    template = ref_gray[band_y:band_y + band_h, :]

    tys = []
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if not ret:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(g[band_y:band_y + band_h, :], template, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        # vertical offset within band
        y_off = int(max_loc[1])
        # map to overall image coordinate (distance from top)
        tys.append(y_off - 0)

    cap.release()

    if not tys:
        raise RuntimeError('Failed to estimate shifts')

    min_ty, max_ty = int(np.min(tys)), int(np.max(tys))
    top_crop = max(0, max_ty)
    bottom_crop = max(0, -min_ty)

    x1, y1, x2, y2 = 0, top_crop, w, max(1, h - bottom_crop)
    box = [x1, y1, x2, y2]
    out = {'constant_crop': box, 'source': 'fallback_template_matching'}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, 'w'))
    print(f'Wrote fallback constant crop: {out_path} -> {box}')
    return box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transforms', type=str, default='', help='Path to numpy .npy transforms dict produced by high-quality stabilizer')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--out', type=str, default='output/constant_crop.json')
    args = parser.parse_args()

    vp = Path(args.video)
    if args.transforms:
        try:
            compute_from_transforms(Path(args.transforms), vp, args.out)
            return
        except Exception as e:
            print('Transforms-based computation failed:', e)
            print('Falling back to estimate')

    estimate_shift_fallback(vp, args.out)


if __name__ == '__main__':
    main()
