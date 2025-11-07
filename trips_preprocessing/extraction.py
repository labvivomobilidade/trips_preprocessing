from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

# Try to use Numba
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    njit = None
    HAVE_NUMBA = False


####################################################################################
# Method 1

def _as_np2(a: List[List[float]]) -> np.ndarray:
    """
    Convert a Python list of [lon, lat] pairs into a NumPy array of shape (N, 2).

    Args:
        a (list[list[float]]): List of [lon, lat] coordinate pairs.

    Returns:
        numpy.ndarray: Array with shape (N, 2) and dtype float64.
    """
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected shape (N, 2) for [lon, lat] pairs.")
    return arr


# ---------------------------------------------------------------------
# 1) Low-level core: simple leave/enter logic (no start/end adjustment)
#    Numba-accelerated version
# ---------------------------------------------------------------------
if HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _segment_trips_leave_enter_core(
        traj: np.ndarray,             # shape (N, 2): [lon, lat]
        start_centers: np.ndarray,    # shape (K, 2): [lon, lat]
        end_centers: np.ndarray,      # shape (K, 2): [lon, lat]
        margin_deg: float,            # half side of the (square) geofence in degrees
        consider_mask: np.ndarray     # shape (N,), bool; if all True, full sequence is considered
    ):
        """
        Detect trip segments based on entering and leaving geofenced areas.

        Args:
            traj (numpy.ndarray): Array of shape (N, 2) containing [lon, lat] coordinates.
            start_centers (numpy.ndarray): Array of shape (K, 2) defining start geofences.
            end_centers (numpy.ndarray): Array of shape (K, 2) defining end geofences.
            margin_deg (float): Half-side of the square geofence in degrees.
            consider_mask (numpy.ndarray): Boolean array (N,) indicating which points to consider.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                Arrays for start indices, end indices, and matching geofence pair indices.
        """
        N = traj.shape[0]
        K = start_centers.shape[0]

        starts: List = []
        ends:   List = []
        pairs:  List = []

        # Track “inside which start geofence” for previous sample
        prev_inside_start_k = -1
        # Track “currently inside which start geofence” for current sample
        curr_inside_start_k = -1

        # Track trip state
        in_trip = False
        target_pair = -1    # which end-center we expect to enter

        # Utility: point-in-square (start) and (end)
        def find_inside_idx(lat_val, lon_val, centers):
            for kk in range(K):
                dlat = abs(lat_val - centers[kk, 0])
                dlon = abs(lon_val - centers[kk, 1])
                if (dlat <= margin_deg) and (dlon <= margin_deg):
                    return kk
            return -1

        for i in range(N):
            if not consider_mask[i]:
                # Even when skipping area logic, we still maintain prev_inside state
                # to avoid false transitions across ignored samples
                prev_inside_start_k = curr_inside_start_k
                continue

            lat_i = traj[i, 0]
            lon_i = traj[i, 1]

            # Are we inside any start geofence now?
            curr_inside_start_k = find_inside_idx(lat_i, lon_i, start_centers)

            if not in_trip:
                # If we were inside start[k] and now we are OUTSIDE it -> START
                if (prev_inside_start_k >= 0) and (curr_inside_start_k == -1):
                    # Pair index is the start geofence we just left
                    target_pair = prev_inside_start_k
                    starts.append(i)        # start at FIRST sample outside
                    ends.append(-1)         # placeholder
                    pairs.append(target_pair)
                    in_trip = True
            else:
                # We are in a trip; check if we ENTER the paired end geofence
                # (square check against end_centers[target_pair])
                lat_c = end_centers[target_pair, 0]
                lon_c = end_centers[target_pair, 1]
                if (abs(lat_i - lat_c) <= margin_deg) and (abs(lon_i - lon_c) <= margin_deg):
                    # END at FIRST sample inside the target end geofence
                    ends[len(ends) - 1] = i
                    in_trip = False
                    target_pair = -1

            prev_inside_start_k = curr_inside_start_k

        # return as NumPy arrays for convenience
        return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64), np.array(pairs, dtype=np.int64)

else:
    # -----------------------------------------------------------------
    # Pure-Python fallback (same logic, no Numba)
    # -----------------------------------------------------------------
    def _segment_trips_leave_enter_core(
        traj: np.ndarray,
        start_centers: np.ndarray,
        end_centers: np.ndarray,
        margin_deg: float,
        consider_mask: np.ndarray
    ):
        """
        Detect trip segments based on entering and leaving geofenced areas (pure Python version).

        Args:
            traj (numpy.ndarray): Array of shape (N, 2) containing [lon, lat] coordinates.
            start_centers (numpy.ndarray): Array of shape (K, 2) defining start geofences.
            end_centers (numpy.ndarray): Array of shape (K, 2) defining end geofences.
            margin_deg (float): Half-side of the square geofence in degrees.
            consider_mask (numpy.ndarray): Boolean array (N,) indicating which points to consider.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                Arrays for start indices, end indices, and matching geofence pair indices.
        """
        N = traj.shape[0]
        K = start_centers.shape[0]

        starts: List[int] = []
        ends:   List[int] = []
        pairs:  List[int] = []

        prev_inside_start_k = -1
        curr_inside_start_k = -1

        in_trip = False
        target_pair = -1

        def find_inside_idx(lat_val, lon_val, centers):
            for kk in range(K):
                dlat = abs(lat_val - centers[kk, 0])
                dlon = abs(lon_val - centers[kk, 1])
                if (dlat <= margin_deg) and (dlon <= margin_deg):
                    return kk
            return -1

        for i in range(N):
            if not bool(consider_mask[i]):
                prev_inside_start_k = curr_inside_start_k
                continue

            lat_i = float(traj[i, 0])
            lon_i = float(traj[i, 1])

            curr_inside_start_k = find_inside_idx(lat_i, lon_i, start_centers)

            if not in_trip:
                if (prev_inside_start_k >= 0) and (curr_inside_start_k == -1):
                    target_pair = prev_inside_start_k
                    starts.append(i)
                    ends.append(-1)
                    pairs.append(target_pair)
                    in_trip = True
            else:
                lat_c = end_centers[target_pair, 0]
                lon_c = end_centers[target_pair, 1]
                if (abs(lat_i - lat_c) <= margin_deg) and (abs(lon_i - lon_c) <= margin_deg):
                    ends[-1] = i
                    in_trip = False
                    target_pair = -1

            prev_inside_start_k = curr_inside_start_k

        return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64), np.asarray(pairs, dtype=np.int64)


# ---------------------------------------------------------------------
# 2) High-level function
# ---------------------------------------------------------------------

def extract_trips(trajectory:pd.DataFrame, start_center:tuple[float], end_center:tuple[float],
                  radius:float = 50, consider_n_points:int = 0) -> list[pd.DataFrame]:
    """
    Extract trip segments from a trajectory based on geofence leave/enter logic.

    Args:
        trajectory (pd.DataFrame): DataFrame containing 'longitude' and 'latitude' columns.
        start_center (tuple[float, float]): Start geofence centers as [lon, lat].
        end_center (tuple[float, float]): End geofence centers as [lon, lat].
        radius (float): Half-side of the square geofence, in meters. Default is 50. Nn the code a conversion is made to degs.
        consider_n_points (int): Consider starting n seconds before and ending n seconds after.
    Returns:
        list[pd.DataFrame]: List of DataFrames, each representing one trip segment.
    """
    margin_deg:float = radius/111_132.95 # Convertion to radius to meters more in (https://www.fws.gov/r7/nwr/Realty/data/LandMappers/Public/Help/HTML/R7-Public-Land-Mapper-Help.html?Degreesandgrounddistance.html)
    latitude, longitude = "latitude", "longitude"
    
    all_trajectory = trajectory.copy()
    trajectory = trajectory[[longitude, latitude]]
    traj = _as_np2(trajectory)
    starts_np = np.asarray([[start_center[0], start_center[1]]], dtype=np.float64)
    ends_np   = np.asarray([[end_center[0],  end_center[1]]],  dtype=np.float64)

    s_idx, e_idx, k_idx = _segment_trips_leave_enter_core(
        traj, starts_np, ends_np, float(margin_deg), np.ones(len(trajectory), dtype=np.bool_)
    )

    trips: List[Dict[str, int]] = []
    for s, e, k in zip(s_idx, e_idx, k_idx):
        if e >= 0 and e >= s:
            trips.append({"start_idx": max(0, int(s) - consider_n_points),
                          "end_idx": min(int(e) + consider_n_points, len(all_trajectory)),
                          "pair_idx": int(k)})

    all_trips:list = []
    for trip in trips:
        all_trips.append(all_trajectory.iloc[trip["start_idx"]:trip["end_idx"]])

    return all_trips

####################################################################################
# Method 2

def _find_operations_py(dt: np.ndarray, dist_deg: np.ndarray,
                        margin_deg: float, window_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    N = dt.shape[0]
    outside_streak = 0.0
    inside_streak  = 0.0
    in_operation   = False

    starts = []
    ends   = []

    for i in range(N):
        outside = dist_deg[i] > margin_deg
        if not in_operation:
            if outside:
                outside_streak += dt[i]; inside_streak = 0.0
            else:
                outside_streak = 0.0;    inside_streak += dt[i]
            if outside_streak >= window_seconds:
                starts.append(i)
                in_operation = True
                inside_streak = 0.0
        else:
            if not outside:
                inside_streak += dt[i]; outside_streak = 0.0
            else:
                inside_streak = 0.0;    outside_streak += dt[i]
            if inside_streak >= window_seconds:
                ends.append(i)
                in_operation = False
                outside_streak = 0.0

    if in_operation:
        ends.append(N - 1)

    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)

if njit is not None:
    @njit(cache=True, fastmath=True)
    def _find_operations_nb(dt, dist_deg, margin_deg, window_seconds):
        N = dt.shape[0]
        starts = np.empty(N, np.int64)
        ends   = np.empty(N, np.int64)
        ns = 0; ne = 0

        outside_streak = 0.0
        inside_streak  = 0.0
        in_operation   = False

        for i in range(N):
            outside = dist_deg[i] > margin_deg
            if not in_operation:
                if outside:
                    outside_streak += dt[i]; inside_streak = 0.0
                else:
                    outside_streak = 0.0;    inside_streak += dt[i]
                if outside_streak >= window_seconds:
                    starts[ns] = i; ns += 1
                    in_operation = True
                    inside_streak = 0.0
            else:
                if not outside:
                    inside_streak += dt[i]; outside_streak = 0.0
                else:
                    inside_streak = 0.0;    outside_streak += dt[i]
                if inside_streak >= window_seconds:
                    ends[ne] = i; ne += 1
                    in_operation = False
                    outside_streak = 0.0

        if in_operation:
            ends[ne] = N - 1; ne += 1

        return starts[:ns], ends[:ne]
else:
    _find_operations_nb = None  # sem numba disponível


def extract_operations_movel_mean(
    df_raw: pd.DataFrame,
    *,
    margin_deg:float = 0.0003,
    window_seconds:int = 45,
    wait_window:int = 90,
    merge_if_samples_leq:int = 90,
    min_len_samples:int = 180,
    max_len_samples:int = 6_000,
    var_threshold:float = 9e-06,
    time_col:str = "timestamp",
    lat_col:str = "latitude",
    lon_col:str = "longitude",
    use_numba:bool = True,
) -> List[pd.DataFrame]:
    """
    Extract operation windows from a lat/lon time series using a moving-average band
    and stay-time rules (outside/inside) to open/close segments.

    Args:
        df_raw (pd.DataFrame): Input with at least:
            - `time_col` (timestamp parseable to datetime),
            - `lat_col` (float latitude),
            - `lon_col` (float longitude).
            Rows may be irregularly sampled; they will be time-sorted.
        margin_deg (float): Band half-width (in degrees) around the time-based moving average.
            Points with Euclidean distance (lat/lon) to the moving average > margin_deg are "outside".
        window_seconds (int): Minimum dwell time (in seconds) required to OPEN an operation
            while "outside" and to CLOSE it while "inside".
        merge_if_samples_leq (int): If a detected operation has length (in samples) ≤ this value,
            it is merged into the next operation.
        min_len_samples (int): Minimum number of samples for an operation to be kept.
        max_len_samples (int): Maximum allowed number of samples for an operation.
        var_threshold (float): Minimum variance required in BOTH `lon_col` and `lat_col`.
            Operations with variance below this in both axes are discarded (near-stationary).
        time_col (str): Name of the timestamp column. Default: "timestamp".
        lat_col (str): Name of the latitude column. Default: "latitude".
        lon_col (str): Name of the longitude column. Default: "longitude".

    Returns:
        list[pd.DataFrame]: Chronologically ordered list of slices from `df_raw`,
            one per detected operation (semi-open intervals `iloc[start:end]`),
            with temporary moving-average columns removed.
    """

    df = df_raw.copy()

    ts = df[time_col]
    if not is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce", utc=True)
    else:
        if is_datetime64tz_dtype(ts):
            ts = ts.dt.tz_convert("UTC")
        else:
            ts = ts.dt.tz_localize("UTC")

    ts = ts.dt.tz_localize(None)
    df[time_col] = ts
    df = df.dropna(subset=[time_col, lat_col, lon_col]).sort_values(time_col).reset_index(drop=True)

    idx = pd.DatetimeIndex(df[time_col])
    lat_mm = pd.Series(df[lat_col].to_numpy(), index=idx).rolling(f"{window_seconds}s").mean().to_numpy()
    lon_mm = pd.Series(df[lon_col].to_numpy(), index=idx).rolling(f"{window_seconds}s").mean().to_numpy()

    lat = df[lat_col].to_numpy(dtype="float64")
    lon = df[lon_col].to_numpy(dtype="float64")

    mm_nan = np.isnan(lat_mm) | np.isnan(lon_mm)
    dlat = lat - lat_mm
    dlon = lon - lon_mm
    dlat[mm_nan] = 0.0
    dlon[mm_nan] = 0.0
    dist_deg = np.hypot(dlat, dlon)

    t_ns = df[time_col].astype("int64").to_numpy()   # ns
    t_sec = t_ns * 1e-9

    dt = np.empty_like(t_sec, dtype="float64")
    dt[0] = 0.0
    np.subtract(t_sec[1:], t_sec[:-1], out=dt[1:])
    dt[dt < 0] = 0.0

    if use_numba and _find_operations_nb is not None:
        starts, ends = _find_operations_nb(dt, dist_deg, float(margin_deg), float(wait_window))
    else:
        starts, ends = _find_operations_py(dt, dist_deg, float(margin_deg), float(wait_window))

    if starts.size and ends.size:
        L = min(starts.size, ends.size)
        starts = starts[:L]; ends = ends[:L]

        keep_starts = []
        keep_ends   = []

        i = 0
        while i < L:
            s = int(starts[i]); e = int(ends[i])
            while (e - s) <= merge_if_samples_leq and (i + 1) < L:
                i += 1
                e = int(ends[i])
            keep_starts.append(s); keep_ends.append(e)
            i += 1

        starts = np.asarray(keep_starts, dtype=np.int64)
        ends   = np.asarray(keep_ends,   dtype=np.int64)

    ops: List[pd.DataFrame] = []
    for s, e in zip(starts, ends):
        length = int(e) - int(s)
        if length < min_len_samples or length > max_len_samples:
            continue

        chunk = df.iloc[s:e].copy()

        if (chunk[lon_col].var() < var_threshold) and (chunk[lat_col].var() < var_threshold):
            continue

        ops.append(chunk)

    return ops
