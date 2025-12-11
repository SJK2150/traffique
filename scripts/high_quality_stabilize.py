"""High-quality stabilization helper.

Estimates per-frame affine transforms using KLT feature tracks, computes
accumulated transforms to a reference, parameterizes (tx,ty,theta,scale),
applies Gaussian smoothing on parameters and writes smoothed cumulative
transforms to disk as a numpy .npy file mapping frame -> 3x3 matrix.

This is a pragmatic, high-quality stabilization step that balances accuracy
and dependency-light implementation (uses OpenCV + NumPy).
"""
from pathlib import Path
import json
import numpy as np
import cv2
import math
import os
from typing import Tuple


def find_video_path(video_name: str, repo_root: Path) -> Path:
    uploads = repo_root / 'uploads'
    candidate = uploads / video_name
    if candidate.exists():
        return candidate
    local = Path(r'C:/Users/sakth/Documents/traffique_footage') / video_name
    if local.exists():
        return local
    if Path(video_name).exists():
        return Path(video_name)
    raise FileNotFoundError(f"Video not found: {video_name}")


def _estimate_affine_klt(video_path: Path, start_frame: int, end_frame: int, max_corners=2000) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Cannot read start frame {start_frame}")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = {}

    # detect initial corners in prev
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=6, blockSize=7)
    if prev_pts is None:
        prev_pts = np.empty((0, 1, 2), dtype=np.float32)

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    for f in range(start_frame + 1, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, cur = cap.read()
        if not ret:
            break
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        if prev_pts is None or len(prev_pts) < 20:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=6, blockSize=7)
            if prev_pts is None:
                prev_pts = np.empty((0, 1, 2), dtype=np.float32)

        if prev_pts.size == 0:
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            transforms[f] = M
            prev_gray = cur_gray
            continue

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_pts, None, **lk_params)
        if next_pts is None:
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            transforms[f] = M
            prev_gray = cur_gray
            continue

        good_prev = prev_pts[status.flatten() == 1].reshape(-1, 2)
        good_next = next_pts[status.flatten() == 1].reshape(-1, 2)

        if len(good_prev) < 6:
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        else:
            M, inliers = cv2.estimateAffinePartial2D(good_next, good_prev, method=cv2.RANSAC, ransacReprojThreshold=4.0)
            if M is None:
                M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        transforms[f] = M.astype(np.float32)

        # advance
        prev_gray = cur_gray
        prev_pts = good_next.reshape(-1, 1, 2).astype(np.float32) if len(good_next) else None

    cap.release()
    return transforms


def _accumulate(transforms: dict, start_frame: int, end_frame: int) -> dict:
    cumulative = {}
    cumulative[start_frame] = np.eye(3, dtype=np.float32)
    for f in range(start_frame + 1, end_frame + 1):
        T = transforms.get(f, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
        T33 = np.eye(3, dtype=np.float32)
        T33[0:2, :] = T
        cumulative[f] = cumulative[f - 1] @ T33
    return cumulative


def _decompose_affine(M33: np.ndarray) -> Tuple[float, float, float, float]:
    # M33 is 3x3
    A = M33[0:2, 0:2]
    tx = float(M33[0, 2])
    ty = float(M33[1, 2])
    # rotation angle
    theta = math.atan2(A[1, 0], A[0, 0])
    # scale (approx)
    scale = math.sqrt(A[0, 0] * A[0, 0] + A[1, 0] * A[1, 0])
    return tx, ty, theta, scale


def _compose_affine_params(tx, ty, theta, scale) -> np.ndarray:
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    S = scale
    A = R * S
    M = np.eye(3, dtype=np.float32)
    M[0:2, 0:2] = A
    M[0, 2] = tx
    M[1, 2] = ty
    return M


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.5:
        return x
    # kernel radius
    radius = int(max(1, math.ceil(3 * sigma)))
    size = radius * 2 + 1
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-(xs ** 2) / (2 * sigma * sigma))
    kernel = kernel / np.sum(kernel)
    # pad edges
    padded = np.pad(x, ((radius, radius), (0, 0)), mode='edge') if x.ndim == 2 else np.pad(x, radius, mode='edge')
    if x.ndim == 1:
        out = np.convolve(padded, kernel, mode='valid')
        return out
    else:
        out = np.empty_like(x)
        for i in range(x.shape[1]):
            out[:, i] = np.convolve(padded[:, i], kernel, mode='valid')
        return out


def high_quality_stabilize(video_path: Path, start_frame: int, end_frame: int, out_dir: Path, sigma: float = 3.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Estimating KLT-based affine transforms for frames {start_frame}..{end_frame}")
    transforms = _estimate_affine_klt(video_path, start_frame, end_frame)
    cumulative = _accumulate(transforms, start_frame, end_frame)

    # collect parameter vectors
    frames = list(range(start_frame, end_frame + 1))
    params = []
    for f in frames:
        M33 = cumulative.get(f, np.eye(3, dtype=np.float32))
        params.append(_decompose_affine(M33))
    params = np.array(params, dtype=float)  # shape (N,4)

    # smooth parameters
    print(f"Smoothing transform parameters (sigma={sigma} frames)")
    params_sm = _gaussian_smooth(params, sigma=sigma)

    # reconstruct smoothed cumulative transforms
    smoothed = {}
    for i, f in enumerate(frames):
        tx, ty, theta, scale = float(params_sm[i, 0]), float(params_sm[i, 1]), float(params_sm[i, 2]), float(params_sm[i, 3])
        M33 = _compose_affine_params(tx, ty, theta, scale)
        smoothed[f] = M33.astype(np.float32)

    # compute safe crop: intersection of warped frame corners
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        h, w = 720, 1280
    else:
        h, w = frame.shape[:2]

    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    # warp corners and find intersection bbox
    all_boxes = []
    for f in frames:
        M = smoothed[f]
        pts = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M)[..., 0, :]
        xs = pts[:, 0]
        ys = pts[:, 1]
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        all_boxes.append((minx, miny, maxx, maxy))

    # intersection
    minx = max(box[0] for box in all_boxes)
    miny = max(box[1] for box in all_boxes)
    maxx = min(box[2] for box in all_boxes)
    maxy = min(box[3] for box in all_boxes)
    safe_box = [int(round(minx)), int(round(miny)), int(round(maxx)), int(round(maxy))]

    # Save outputs
    trans_out = out_dir / f"smoothed_transforms_{start_frame}_{end_frame}.npy"
    np.save(str(trans_out), smoothed, allow_pickle=True)
    json.dump({'safe_box': safe_box, 'start_frame': start_frame, 'end_frame': end_frame}, open(out_dir / 'safe_crop.json', 'w'))
    print(f"Wrote smoothed transforms: {trans_out}")
    return trans_out, out_dir / 'safe_crop.json'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument('--video', type=str, default='D1F1_stab.mp4')
    parser.add_argument('--frame', type=int, default=9861)
    parser.add_argument('--time_window', type=int, default=15)
    parser.add_argument('--out', type=str, default=str(Path(__file__).resolve().parents[1] / 'output' / 'stabilize_cache'))
    parser.add_argument('--sigma', type=float, default=3.0)
    args = parser.parse_args()
    video_path = find_video_path(args.video, repo_root)
    fps = cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS) or 25.0
    start = int(args.frame)
    end = int(args.frame + max(1, int(args.time_window * fps)))
    high_quality_stabilize(video_path, start, end, Path(args.out), sigma=args.sigma)
