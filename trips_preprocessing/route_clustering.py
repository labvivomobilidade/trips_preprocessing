from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional


def compare_routes_kdtree(r1:pd.DataFrame, r2:pd.DataFrame, truncation_deg:float = 0.0001, lat_col:str = "latitude", lon_col:str = "longitude", scale:float = 1000.0) -> float:
    """
    Compute a similarity score (in [0, 1]) between two routes.

    Inputs:
        r1, r2 : pd.DataFrame
            DataFrames for the two routes, each with 'lat_col' and 'lon_col' columns.
        truncation_deg : float
            Truncation step in degrees (e.g., 0.0001). Helps reduce noise and cost.
        lat_col, lon_col : str
            Column names for latitude and longitude.
        scale : float
            Scaling factor to convert distance into a penalty in the score (default ~1000).

    Output:
        float
            Similarity ∈ [0, 1]; higher values indicate more similar (closer) routes.
    """
    def _truncate(arr: pd.Series, step: float) -> np.ndarray:
        factor = 1.0 / step
        return np.trunc(arr.to_numpy() * factor) / factor

    # Verificações básicas
    for col in (lat_col, lon_col):
        if col not in r1.columns or col not in r2.columns:
            raise ValueError(f"Coluna '{col}' ausente em uma das rotas.")

    r1c = r1[[lat_col, lon_col]].copy()
    r2c = r2[[lat_col, lon_col]].copy()

    r1c[lat_col] = _truncate(r1c[lat_col], truncation_deg)
    r1c[lon_col] = _truncate(r1c[lon_col], truncation_deg)
    r2c[lat_col] = _truncate(r2c[lat_col], truncation_deg)
    r2c[lon_col] = _truncate(r2c[lon_col], truncation_deg)

    r1c = r1c.drop_duplicates().dropna()
    r2c = r2c.drop_duplicates().dropna()

    if r1c.empty or r2c.empty:
        return 0.0

    tree1 = cKDTree(r1c[[lat_col, lon_col]].to_numpy())
    tree2 = cKDTree(r2c[[lat_col, lon_col]].to_numpy())

    d12, _ = tree2.query(r1c[[lat_col, lon_col]].to_numpy(), k = 1)  # r1 -> r2
    d21, _ = tree1.query(r2c[[lat_col, lon_col]].to_numpy(), k = 1)  # r2 -> r1

    s1 = np.mean(1.0 / (1.0 + scale * d12))
    s2 = np.mean(1.0 / (1.0 + scale * d21))

    return float(np.sqrt(s1 * s2))

def build_similarity_matrix(routes:List[pd.DataFrame], truncation_deg:float = 0.0001, lat_col:str = "latitude", lon_col:str = "longitude",
                            scale:float = 1000.0, combinations:Optional[int] = None, random_state:Optional[int] = None, progress:bool = False) -> np.ndarray:
    """
    Build an NxN similarity matrix in [0, 1] across routes.

    - If `combinations` is None: compute ALL pairwise similarities (full matrix).
    - If `combinations` is an integer k: for each i, sample k distinct j’s (partial matrix).
      Pairs not evaluated remain 0 (later treated as maximum distance).

    Inputs:
        routes : list[pd.DataFrame]
            List of routes; each DataFrame must contain latitude/longitude columns.
        truncation_deg, lat_col, lon_col, scale : from compare_routes_kdtree.
        combinations : Optional[int]
            Number of random j’s per row i (partial mode). None = full matrix.
        random_state : Optional[int]
            RNG seed for reproducibility.
        progress : bool
            True to print simple progress.

    Output:
        np.ndarray
            NxN similarity matrix (float), with diagonal = 1.0. In partial mode,
            unevaluated entries remain 0.
    """
    n:int = len(routes)
    S = np.zeros((n, n), dtype = float)
    np.fill_diagonal(S, 1.0)

    rng = np.random.default_rng(random_state)

    if combinations is None:
        total = n * (n - 1) // 2
        done = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                sim:float = compare_routes_kdtree(routes[i], routes[j], truncation_deg = truncation_deg,
                                                  lat_col = lat_col, lon_col = lon_col, scale = scale)
                S[i, j] = S[j, i] = sim
                done += 1
                if progress and done % 50 == 0:
                    print(f"\rPares computados: {done}/{total}", end="")
        if progress:
            print(f"\rPares computados: {total}/{total}")
    else:
        k = max(0, min(combinations, n - 1))
        for i in range(n):
            candidates = [j for j in range(n) if j != i]
            if k > 0:
                js = rng.choice(candidates, size=k, replace=False)
            else:
                js = []
            for j in js:
                if S[i, j] > 0 or S[j, i] > 0:
                    continue
                sim:float = compare_routes_kdtree(routes[i], routes[j], truncation_deg = truncation_deg,
                                                  lat_col = lat_col, lon_col = lon_col, scale=scale)
                S[i, j] = S[j, i] = sim
            if progress and (i % 10 == 0):
                print(f"\rLinhas processadas: {i+1}/{n}", end = "")
        if progress:
            print(f"\rLinhas processadas: {n}/{n}")

    return S

_GLOBAL_ROUTES: List[pd.DataFrame] = []
_GLOBAL_TRUNC: float = 0.0001
_GLOBAL_LAT: str = "latitude"
_GLOBAL_LON: str = "longitude"
_GLOBAL_SCALE: float = 1000.0

def _init_pool(routes: List[pd.DataFrame],
               truncation_deg: float,
               lat_col: str,
               lon_col: str,
               scale: float) -> None:
    global _GLOBAL_ROUTES, _GLOBAL_TRUNC, _GLOBAL_LAT, _GLOBAL_LON, _GLOBAL_SCALE
    _GLOBAL_ROUTES = routes
    _GLOBAL_TRUNC = truncation_deg
    _GLOBAL_LAT = lat_col
    _GLOBAL_LON = lon_col
    _GLOBAL_SCALE = scale

def _worker_pair(pair: Tuple[int, int]) -> Tuple[int, int, float]:
    i, j = pair
    sim = compare_routes_kdtree(
        _GLOBAL_ROUTES[i],
        _GLOBAL_ROUTES[j],
        truncation_deg=_GLOBAL_TRUNC,
        lat_col=_GLOBAL_LAT,
        lon_col=_GLOBAL_LON,
        scale=_GLOBAL_SCALE,
    )
    return i, j, sim

def build_similarity_matrix_pool(
    routes: List[pd.DataFrame],
    truncation_deg: float = 0.0001,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    scale: float = 1000.0,
    progress: bool = False,
    cpus: Optional[int] = None,
    chunksize: int = 256,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Build the NxN similarity matrix using multiprocessing.Pool.

    Strategy:
        - Store 'routes' in process-global variables via an initializer to avoid
        repeated serialization overhead.
        - Generate the list of index pairs (i, j) with i < j.
        - Use Pool.imap_unordered(_worker_pair, ...) with a tuned chunksize to
        balance parallel overhead and throughput.
        - Fill the similarity matrix symmetrically (S[i, j] = S[j, i]).

    Notes:
        - On Linux (fork), memory pages are inherited efficiently; on Windows (spawn),
        pickling overhead is higher.
        - Adjust chunksize based on the cost per pair: typically in the hundreds or
        thousands for best throughput.
    """

    rng = np.random.default_rng(random_state)
    n = len(routes)
    S = np.zeros((n, n), dtype=float)
    np.fill_diagonal(S, 1.0)

    pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    total = len(pairs)

    if cpus is None:
        cpus = max(1, (cpu_count() or 4) - 1)

    with Pool(
        processes=cpus,
        initializer=_init_pool,
        initargs=(routes, truncation_deg, lat_col, lon_col, scale),
        maxtasksperchild=None,
    ) as pool:
        done = 0
        for i, j, sim in pool.imap_unordered(_worker_pair, pairs, chunksize=chunksize):
            S[i, j] = S[j, i] = sim
            done += 1
            if progress and (done % 500 == 0 or done == total):
                print(f"\rpares computados: {done}/{total}", end="")

    if progress:
        print(f"\rpares computados: {total}/{total}")

    return S

def cluster_routes_from_similarity(S:np.ndarray, cut_distance:float = 0.04, method:str = "average") -> Dict[int, List[int]]:
    """
        Build clusters from a similarity matrix S ∈ [0, 1].
        Converts S into a distance matrix D = 1 − S, performs hierarchical linkage,
        and applies a distance cut via fcluster. Returns a dict label -> indices.

        Inputs:
            S: np.ndarray
                NxN similarity matrix (ones on the diagonal). Entries equal to 0 are
                treated as distance 1 (unknown = maximum dissimilarity).
            cut_distance: float
                Distance threshold used to cut the dendrogram (criterion="distance").
            method: str
                Linkage method (e.g., "average", "single", "complete", "ward"
                — not recommended when using D = 1 − S).

        Outputs:
            dict[int, list[int]]: Mapping {group_id: [route_indices]}.
            dict[int, int]:      Mapping {group_id: representative_index}.
    """

    S = np.asarray(S, dtype = float)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)

    D_condensed = squareform(D, checks = False)

    Z = linkage(D_condensed, method = method)
    labels = fcluster(Z, t = cut_distance, criterion = "distance")

    groups = defaultdict(list)
    for idx, g in enumerate(labels):
        groups[int(g)].append(int(idx))

    groups_sorted: Dict[int, List[int]] = dict(
        sorted(groups.items(), key = lambda kv: (-len(kv[1]), kv[0]))
    )

    group_medoids: Dict[int, int] = {}
    for g, idxs in groups_sorted.items():
        if len(idxs) == 1:
            group_medoids[g] = idxs[0]
            continue
        subD = D[np.ix_(idxs, idxs)]
        row_scores = np.sum(subD**2, axis = 1)
        medoid_local_pos = int(np.argmin(row_scores))
        group_medoids[g] = idxs[medoid_local_pos]

    return groups_sorted, group_medoids


def cluster_routes(routes:List[pd.DataFrame], truncation_deg:float = 0.0001, lat_col:str = "latitude", lon_col:str = "longitude",
                   scale:float = 1000.0, combinations:Optional[int] = None, random_state:Optional[int] = None, progress:bool = False,
                   cut_distance:float = 0.04, method:str = "average", cpus:int = 1)  -> (Dict[int, List[int]], Dict[int, int]):
    """
    Inputs:
        routes: list[pd.DataFrame]
            List of routes; each DataFrame must contain lat/lon columns.
        truncation_deg, lat_col, lon_col, scale: forwarded from compare_routes_kdtree.
        combinations: Optional[int]
            Number of random pairs per row (partial). None = evaluate all pairs.
        random_state: Optional[int]
            Seed for the RNG to ensure reproducibility.
        progress: bool
            True to print simple progress updates.
        cut_distance: float
            Cut threshold in the dendrogram (using the "distance" criterion).
        method: str
            Linkage method (e.g., "average", "single", "complete", "ward").

    Outputs:
        dict[int, list[int]]: Mapping {group_id: [route_indices]}.
        dict[int, int]:      Mapping {group_id: representative_index}.
    """

    if cpus == 1:
        S = build_similarity_matrix(routes = routes, truncation_deg = truncation_deg, lat_col = lat_col, lon_col = lon_col,
                                    scale = scale, combinations = combinations, random_state = random_state, progress = progress)
    else:
        S = build_similarity_matrix_pool(routes = routes, truncation_deg = truncation_deg, lat_col = lat_col, lon_col = lon_col,
                                         scale = scale, random_state = random_state, progress = progress, cpus = cpus)

    return cluster_routes_from_similarity(S, cut_distance = cut_distance, method = method)