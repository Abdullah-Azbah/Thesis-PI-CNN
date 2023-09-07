import numpy as np
from dataclasses import dataclass


@dataclass
class Results:
    nodes_number: np.ndarray
    nodes_location: np.ndarray
    stress_intensity: np.ndarray
    nodal_displacement_x: np.ndarray
    nodal_displacement_y: np.ndarray

    # normal_stress_x
    # normal_stress_y
    # shear_stress_xy

    # principal_stress_1
    # principal_stress_3

