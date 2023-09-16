import numpy as np
from dataclasses import dataclass


@dataclass
class Results:
    nodes_number: np.ndarray
    nodes_location: np.ndarray
    nodal_displacement_x: np.ndarray
    nodal_displacement_y: np.ndarray

