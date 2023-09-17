import pickle
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class FieldMatrices:
    width: float
    height: float
    resolution: float

    applied_displacement_x: np.ndarray
    applied_displacement_y: np.ndarray
    applied_force_x: np.ndarray
    applied_force_y: np.ndarray
    elasticity: np.ndarray

    displacement_x: np.ndarray
    displacement_y: np.ndarray

    n_fields = 7

    def plot_matrix(self, matrix, title):
        fig, ax = plt.subplots(1, 1)
        ax.set(aspect='equal')

        cont = ax.imshow(matrix, interpolation=None,  origin='lower')
        ax.set_title(f'Field: {title}')
        fig.colorbar(cont)
        fig.show()

    def plot_field_matrix(self, field):
        matrix = getattr(self, field)
        self.plot_matrix(matrix, field)

    def plot_field_matrices(self, resolution=None):
        if not resolution:
            resolution = self.resolution

        matrix_shape = (self.height // resolution + 1, self.width // resolution + 1)

        plot_start_x = 0 - matrix_shape[1] // 10
        plot_end_x = matrix_shape[1] + matrix_shape[1] // 10
        plot_start_y = 0 - matrix_shape[0] // 10
        plot_end_y = matrix_shape[0] + matrix_shape[0] // 10

        plots = []
        fig, axs = plt.subplots(2, 4)
        plots.append(axs[0, 0].contourf(self.applied_force_x))
        plots.append(axs[0, 1].contourf(self.applied_force_y))
        plots.append(axs[0, 2].contourf(self.applied_displacement_x))
        plots.append(axs[0, 3].contourf(self.applied_displacement_y))

        plots.append(axs[1, 0].contourf(self.elasticity))

        plots.append(axs[1, 1].contourf(self.displacement_x))
        plots.append(axs[1, 2].contourf(self.displacement_y))

        axs = axs.reshape(-1)
        for i in range(self.n_fields):
            axs[i].set(
                xlim=[plot_start_x, plot_end_x],
                ylim=[plot_start_y, plot_end_y],
                aspect='equal'
            )
            fig.colorbar(plots[i])
        fig.set_dpi(800)
        fig.set_size_inches(16, 4)
        fig.tight_layout()
        fig.show()

    def get_input_matrix(self):
        return np.stack([
            self.applied_force_y,
            self.applied_displacement_y,
            self.elasticity
        ], -1)

    def get_output_matrix(self):
        return np.stack([
            self.displacement_x,
            self.displacement_y,
        ], -1)

    def save(self, save_path):
        with open(save_path, 'wb') as o:
            pickle.dump(self, o)

    @staticmethod
    def from_file(file_path):
        with open(file_path, 'rb') as i:
            matrices = pickle.load(i)
            return matrices
