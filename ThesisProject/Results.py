import numpy as np
from dataclasses import dataclass


@dataclass
class Results:
    nodes_number: np.ndarray
    nodes_location: np.ndarray
    nodal_strain_xx: np.ndarray
    nodal_strain_yy: np.ndarray
    nodal_strain_xy: np.ndarray

