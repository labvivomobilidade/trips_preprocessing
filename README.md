# trips_preprocessing

Utilities for detecting and segmenting movement events (trips or operational windows) from GPS-like latitude/longitude time series.
Includes two main methods:
- Extraction of **trips** using **geofence leave/enter logic**.
- Extraction of **operation windows** based on **moving-average deviation** over time.

---

## Environment Setup

### Requirements
- Python >= 3.9

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

- **1.0.0**
    - First usable and downloadble version.

# Use

## Functions

### extract_trips

[]

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