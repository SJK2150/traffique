"""Simple trajectory utilities: interpolation and smoothing.

These functions provide minimal behavior used by the analysis pipeline so
we don't require scipy/advanced packages for a quick test.
"""
from typing import List, Tuple
import numpy as np


def linear_interpolate(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Linearly interpolate over None values in a list of (x,y) or None.

    Returns a list with no None values.
    """
    pts = [None if p is None else (float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    xs = [p[0] if p is not None else None for p in pts]
    ys = [p[1] if p is not None else None for p in pts]

    def interp(arr):
        out = arr[:]
        # find indices of valid
        valid = [i for i, v in enumerate(arr) if v is not None]
        if not valid:
            return [0.0] * n
        for i in range(n):
            if out[i] is None:
                # find nearest valid left and right
                left = max([j for j in valid if j < i], default=None)
                right = min([j for j in valid if j > i], default=None)
                if left is None:
                    out[i] = out[right]
                elif right is None:
                    out[i] = out[left]
                else:
                    t = (i - left) / (right - left)
                    out[i] = (1 - t) * out[left] + t * out[right]
        return out

    xs_f = interp(xs)
    ys_f = interp(ys)
    return list(zip(xs_f, ys_f))


def median_filter(points: List[Tuple[float, float]], k: int = 3) -> List[Tuple[float, float]]:
    """Apply a simple median filter over sliding window size k (odd).
    """
    if k <= 1:
        return points
    arr = np.array(points, dtype=float)
    n = len(points)
    half = k // 2
    out = []
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        window = arr[l:r]
        med = np.median(window, axis=0)
        out.append((float(med[0]), float(med[1])))
    return out


def smooth_kalman(points: List[Tuple[float, float]], dt: float = 1.0, process_var: float = 1.0, meas_var: float = 4.0) -> List[Tuple[float, float]]:
    """Lightweight 2D Kalman-like smoother.

    This implements a scalar per-dimension Kalman filter (constant-position
    model) with process noise `process_var` and measurement noise `meas_var`.
    It accepts `dt` for API compatibility though the simple filter here does
    not explicitly use `dt` in state propagation (kept for future extension).
    """
    # Handle both numpy arrays and lists
    if isinstance(points, np.ndarray):
        points = points.tolist()
    
    if len(points) == 0:
        return []

    # Ensure no None values; perform simple linear interpolation first
    pts = np.asarray(linear_interpolate(points), dtype=float)

    # State estimate (position) and covariance per-dimension
    x = pts[0].astype(float).copy()
    P = np.array([1.0, 1.0], dtype=float)
    Q = float(process_var)
    R = float(meas_var)

    out = [ (float(x[0]), float(x[1])) ]

    for z in pts[1:]:
        z = np.asarray(z, dtype=float)
        # Predict step: for constant-position model, x remains the same
        P = P + Q

        # Kalman gain
        K = P / (P + R)

        # Update step
        x = x + K * (z - x)

        # Update covariance
        P = (1.0 - K) * P

        out.append((float(x[0]), float(x[1])))

    return out


def rts_smoother(points: List[Tuple[float, float]], dt: float = 1.0, process_var: float = 1.0, meas_var: float = 4.0):
    """Rauch–Tung–Striebel smoother for 2D position+velocity.

    Simple constant-velocity model.
    Returns list of smoothed (x,y) positions.
    """
    if not points:
        return []

    pts = np.asarray(linear_interpolate(points), dtype=float)
    n = len(pts)

    # state: [x, y, vx, vy]
    x = np.zeros((4,))
    x[0], x[1] = pts[0]
    x[2], x[3] = 0.0, 0.0

    # Covariances
    P = np.eye(4) * 1.0
    Q = np.diag([0.25 * process_var, 0.25 * process_var, process_var, process_var])
    R = np.eye(2) * meas_var

    # State transition
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    xs = np.zeros((n, 4))
    Ps = np.zeros((n, 4, 4))

    # forward pass (Kalman filter)
    xs[0] = x.copy()
    Ps[0] = P.copy()
    for i in range(1, n):
        # predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # update with measurement z
        z = pts[i]
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        y = z - (H @ x_pred)
        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred

        xs[i] = x.copy()
        Ps[i] = P.copy()

    # backward pass (RTS)
    x_smooth = xs.copy()
    P_smooth = Ps.copy()
    for i in range(n - 2, -1, -1):
        P_pred = F @ Ps[i] @ F.T + Q
        C = Ps[i] @ F.T @ np.linalg.inv(P_pred)
        x_smooth[i] = xs[i] + C @ (x_smooth[i + 1] - (F @ xs[i]))
        P_smooth[i] = Ps[i] + C @ (P_smooth[i + 1] - P_pred) @ C.T

    out = [(float(x_smooth[i, 0]), float(x_smooth[i, 1])) for i in range(n)]
    return out


if __name__ == "__main__":
    pts = [(0, 0), (1, 1), (2, 2), (100, 100), (4, 4)]
    print(median_filter(pts, k=3))
    print(smooth_kalman(pts, alpha=0.5))
