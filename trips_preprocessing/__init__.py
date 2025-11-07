"""
High-level tools for extracting trip and operation windows from GPS-like
trajectories, and for clustering routes based on geometric similarity.
"""

from .extraction import extract_trips, extract_trips_moving_average
from .route_clustering import cluster_routes

__version__ = "0.0.4"