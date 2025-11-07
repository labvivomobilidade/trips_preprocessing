# trips_preprocessing

Utilities for detecting and segmenting trips from GPS-like latitude/longitude time series.

---

## Environment Setup

### Requirements
- Python 3

---

# Environment

## Install

```
pip install git+https://github.com/labvivomobilidade/trips_preprocessing
```

## Unisnstall

```
uninstall trips_preprocessing
```

## Update

```
pip install --upgrade git+https://github.com/labvivomobilidade/trips_preprocessing
```

or

```
uninstall trips_preprocessing
pip install git+https://github.com/labvivomobilidade/trips_preprocessing
```

# Updates description

- **0.0.1**
    - Preparation of files.

- **0.0.2**
    - Change input of argument `start_center` and `end_center` in `extract_trips` in `[lon, lat]` to `[lat, lon]`.
    - Preparing to make the library installable. 

- **0.0.3**
    - Change name of function. 

- **0.0.4**
    - Add cluster_routes.

- **1.0.0**
    - First usable and downloadble version.

# Use

## Functions

### extract_trips

Extract trip segments using **geofence leave/enter** logic.

**Inputs**
- `trajectory` (pd.DataFrame): Must contain at least `latitude` and `longitude`. Other columns are preserved.
- `start_center` (tuple[float, float]): Center of the **start** geofence, given as `(lat, lon)`.
- `end_center` (tuple[float, float]): Center of the **end** geofence, given as `(lat, lon)`.
- `radius` (float, default=50): Half-side of the square geofence in **meters** (converted internally to degrees).
- `consider_n_points` (int, default=0): Extends each trip segment backward and forward by N samples.

**Output**
- `list[pd.DataFrame]`: Each element is one trip (slice of the original DataFrame).

**Description**
A trip starts when the trajectory **leaves** the start geofence and ends when it **enters** the end geofence.  
The resulting segments maintain all columns from the original data.


---

### extract_trips_moving_average

Extract **movement/operation windows** based on **deviation from a moving average** over time.

**Inputs**
- `df_raw` (pd.DataFrame): Must contain timestamp + latitude + longitude.
- `margin_deg` (float, default=0.0003): Allowed distance (deg) from the moving average before considered “outside.”
- `window_seconds` (int, default=45): Rolling time window (seconds) used to compute the moving average.
- `wait_window` (int, default=90): Time required **outside** to open a window, and **inside** to close it.
- `merge_if_samples_leq` (int, default=90): Merge very short windows into the next one.
- `min_len_samples` / `max_len_samples`: Keep only windows within sample-length bounds.
- `var_threshold` (float, default=9e-06): Discard windows with very low variance (near stationary).
- `time_col`, `lat_col`, `lon_col`: Column names.
- `use_numba` (bool): Accelerates detection when numba is installed.

**Output**
- `list[pd.DataFrame]`: Each element is one detected continuous movement window.

**Description**
Segments are detected when the trajectory **stays away** from its moving-average path for long enough, then returns.  
Useful for identifying activity periods, workstation sessions, etc.


---

### cluster_routes

Cluster multiple routes based on **pairwise geometric similarity**.

**Inputs**
- `routes` (list[pd.DataFrame]): List of route DataFrames.
- `truncation_deg` (float, default=0.0001): Spatial quantization step to reduce noise and computation.
- `lat_col`, `lon_col`: Column names.
- `scale` (float, default=1000.0): Controls how distance affects similarity scores.
- `combinations` (int or None): If `None`, uses all pairs; if integer, samples comparisons.
- `random_state` (optional): Seed for sampling.
- `progress` (bool): Print progress logs.
- `cut_distance` (float, default=0.04): Dendrogram cut threshold for grouping.
- `method` (str, default="average"): Hierarchical linkage method.
- `cpus` (int, default=1): `1` = serial; `>1` uses multiprocessing.

**Output**
- `dict[int, list[int]]`: `{cluster_id: [list of route indices]}`.
- `dict[int, int]`: `{cluster_id: representative_index}` (cluster medoid).

**Description**
Builds a similarity matrix between routes, converts to distance, performs hierarchical clustering, and returns groups and their representative route.

---

## Example

### extract_trips

```python
import pandas as pd
from trips_preprocessing import extract_trips

df = pd.read_csv("trajectory.csv")  # must contain 'latitude' and 'longitude'

start = (12.34567, -45.67890)  # [lat, lon]
end   = (12.34999, -45.68123)  # [lat, lon]

trips = extract_trips(df, start, end, radius=60)

for i, trip in enumerate(trips):
    print(f"Trip {i}: {len(trip)} points")
```

### extract_trips_moving_average

```python
import pandas as pd
from trips_preprocessing import extract_trips_moving_average

df = pd.read_csv("trajectory.csv")  # must contain 'latitude', 'longitude' and 'timestamp'

trips = extract_trips_moving_average(df)

for i, trip in enumerate(trips):
    print(f"Trip {i}: {len(trip)} points")
```

### cluster_routes

```python
import pandas as pd
from trips_preprocessing import extract_trips

df = pd.read_csv("trajectory.csv")  # must contain 'latitude' and 'longitude'

start = (12.34567, -45.67890)  # [lat, lon]
end   = (12.34999, -45.68123)  # [lat, lon]

trips = extract_trips(df, start, end, radius=60)

groups, representants = cluster_routes(trips, cpus = 1)

for group in groups.keys():
    print(f"Group({group}): {groups[group]} <- index of trip in trips")

```