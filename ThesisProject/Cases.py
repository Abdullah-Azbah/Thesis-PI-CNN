import random
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from scipy.interpolate import LinearNDInterpolator
from matplotlib.patches import Rectangle
from ThesisProject.util_functions import round_nearest
from ThesisProject.Results import Results
from ThesisProject.FieldMatrices import FieldMatrices


class Case:
    def save(self, save_path):
        with open(save_path, 'wb') as o:
            pickle.dump(self, o)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, 'rb') as i:
            case: cls = pickle.load(i)
            return case


@dataclass
class CaseV1(Case):
    width: float
    height: float
    left_support_start: float
    left_support_length: float
    right_support_start: float
    right_support_length: float
    force_start: float
    force_length: float
    pressure_magnitude: float
    modulus_of_elasticity: float

    version: str = 'v1'
    resolution: float = 10
    element_size: float = 50
    results: Results = None

    def plot_case(self):
        fig, ax = plt.subplots(1, 1)
        ax.set(aspect='equal')

        ax.add_patch(Rectangle((0, 0), self.width, self.height))
        ax.plot([self.left_support_start, self.left_support_start + self.left_support_length], [0, 0], color='red')
        ax.plot([self.right_support_start, self.right_support_start + self.right_support_length], [0, 0],
                color='red')
        ax.plot([self.force_start, self.force_start + self.force_length], [self.height, self.height], color='green')
        ax.set(xlim=[-500, self.width + 500], ylim=[-500, self.height + 500])
        ax.set_title('Case Input')
        fig.show()

    def plot_result_component(self, component='nodal_strain_xx'):
        fig, ax = plt.subplots(1, 1)
        ax.set(aspect='equal')

        data = getattr(self.results, component)

        x = self.results.nodes_location[:, 0]
        y = self.results.nodes_location[:, 1]
        cont = ax.tricontourf(x, y, data)
        ax.set_title(f'Result: {component}')
        fig.colorbar(cont)
        fig.show()

    @staticmethod
    def random(seed, resolution):
        rng = random.Random()
        rng.seed(seed)

        # width = round_nearest(rng.randint(2500, 5000), resolution)
        # height = round_nearest(rng.randint(width // 4, width), resolution)

        width = 5000
        height = 2500

        left_support_start = round_nearest(rng.uniform(resolution, width / 8), resolution)
        left_support_length = round_nearest(rng.uniform(width / 8, width / 4 - left_support_start), resolution)

        right_support_start = round_nearest(rng.uniform(width * 6 / 8, width * 7 / 8), resolution)
        right_support_length = round_nearest(
            rng.uniform(width / 8 - resolution, width - right_support_start - resolution), resolution)

        force_start = round_nearest(rng.uniform(resolution, 0.75 * width), resolution)
        force_length = round_nearest(rng.uniform(width - force_start - 2 * resolution, width / 4), resolution)

        pressure_magnitude = rng.uniform(1000, 10000)

        min_strain = 1 / height
        max_strain = 10 / height
        strain = rng.uniform(min_strain, max_strain)

        modulus_of_elasticity = pressure_magnitude / strain

        return CaseV1(
            width,
            height,
            left_support_start,
            left_support_length,
            right_support_start,
            right_support_length,
            force_start,
            force_length,
            pressure_magnitude,
            modulus_of_elasticity,

            resolution=resolution
        )

    def create_matrices(self, resolution=None, feature_thickness=200) -> FieldMatrices:
        if not resolution:
            resolution = self.resolution

        feature_thickness = feature_thickness // resolution + 1

        matrix_rows = self.height // resolution + 1
        matrix_cols = self.width // resolution + 1

        x_field = np.arange(0, self.width + 1, resolution)
        y_field = np.arange(0, self.height + 1, resolution)

        matrix_shape = (matrix_rows, matrix_cols)

        left_support_start = self.left_support_start // resolution
        left_support_end = (self.left_support_start + self.left_support_length) // resolution
        right_support_start = self.right_support_start // resolution
        right_support_end = (self.right_support_start + self.right_support_length) // resolution

        applied_displacement_x = np.ones(matrix_shape)
        applied_displacement_x[0: feature_thickness, 0:feature_thickness] = 0
        # applied_displacement_x[0, left_support_start:left_support_end + 1] = 0
        # applied_displacement_x[0, right_support_start:right_support_end + 1] = 0

        applied_displacement_y = np.ones(matrix_shape)
        applied_displacement_y[0: feature_thickness, left_support_start:left_support_end + 1] = 0
        applied_displacement_y[0:feature_thickness, right_support_start:right_support_end + 1] = 0

        elasticity = np.ones(matrix_shape) * self.modulus_of_elasticity

        force_start = self.force_start // resolution
        force_send = (self.force_start + self.force_length) // resolution

        applied_force_x = np.zeros(matrix_shape)

        applied_force_y = np.zeros(matrix_shape)
        applied_force_y[matrix_rows - feature_thickness:matrix_rows,
        force_start:force_send + 1] = self.pressure_magnitude

        x_field_tile = np.tile(x_field, y_field.shape[0]).reshape((-1))
        y_field_tile = np.repeat(y_field, x_field.shape[0]).reshape((-1))

        points = np.empty((matrix_shape[0] * matrix_shape[1], 2))
        points[:, 0] = x_field_tile
        points[:, 1] = y_field_tile

        node_coords = self.results.nodes_location[:, 0:2]

        strain_xx_interpolator = LinearNDInterpolator(node_coords, self.results.nodal_strain_xx)
        strain_field_xx = strain_xx_interpolator(points).reshape(matrix_shape)

        strain_yy_interpolator = LinearNDInterpolator(node_coords, self.results.nodal_strain_yy)
        strain_field_yy = strain_yy_interpolator(points).reshape(matrix_shape)

        strain_xy_interpolator = LinearNDInterpolator(node_coords, self.results.nodal_strain_xy)
        strain_field_xy = strain_xy_interpolator(points).reshape(matrix_shape)

        # plt.imshow(displacement_y_matrix)
        # plt.colorbar()
        # plt.show()
        return FieldMatrices(
            self.width,
            self.height,
            resolution,

            applied_displacement_x,
            applied_displacement_y,
            applied_force_x,
            applied_force_y,

            elasticity,

            strain_field_xx,
            strain_field_yy,
            strain_field_xy
        )

    def analyze(self, apdl_instance):
        apdl_instance.clear("NOSTART")
        apdl_instance.prep7()
        apdl_instance.units("SI")
        apdl_instance.view(xv=0, yv=0, zv=1)

        apdl_instance.et(ename='plane182')
        apdl_instance.mp('ex', 1, self.modulus_of_elasticity)
        apdl_instance.mp('prxy', 1, 0.3)

        k1 = apdl_instance.k(x=0, y=0)

        k2 = apdl_instance.k(x=self.left_support_start, y=0)
        k3 = apdl_instance.k(x=self.left_support_start + self.left_support_length, y=0)

        k4 = apdl_instance.k(x=self.right_support_start, y=0)
        k5 = apdl_instance.k(x=self.right_support_start + self.right_support_length, y=0)

        k6 = apdl_instance.k(x=self.width, y=0)

        k7 = apdl_instance.k(x=0, y=self.height)

        k8 = apdl_instance.k(x=self.force_start, y=self.height)
        k9 = apdl_instance.k(x=self.force_start + self.force_length, y=self.height)

        k10 = apdl_instance.k(x=self.width, y=self.height)

        apdl_instance.l(k1, k2)
        left_support_line = apdl_instance.l(k2, k3)
        apdl_instance.l(k3, k4)
        right_support_line = apdl_instance.l(k4, k5)
        apdl_instance.l(k5, k6)

        apdl_instance.l(k7, k8)
        pressure_line = apdl_instance.l(k8, k9)
        apdl_instance.l(k9, k10)

        apdl_instance.l(k1, k7)
        apdl_instance.l(k6, k10)

        apdl_instance.lsel('all')
        apdl_instance.lesize('all', size=self.element_size)
        apdl_instance.al('all')
        apdl_instance.amesh('all')
        # apdl_instance.eplot('all')

        apdl_instance.run('/solu')
        apdl_instance.time(1)
        apdl_instance.dk(k1, 'ux', 0)
        apdl_instance.dl(left_support_line, lab='UY', value1=0)
        apdl_instance.dl(right_support_line, lab='UY', value1=0)

        apdl_instance.sfl(pressure_line, 'pres', self.pressure_magnitude)
        apdl_instance.allsel('all')

        apdl_instance.solve()
        nnum = apdl_instance.mesh.nnum
        nloc = apdl_instance.mesh.nodes

        apdl_instance.post1()
        apdl_instance.set('last')

        nodal_stain_xx = apdl_instance.post_processing.nodal_elastic_component_strain('X')
        nodal_stain_yy = apdl_instance.post_processing.nodal_elastic_component_strain('Y')
        nodal_stain_xy = apdl_instance.post_processing.nodal_elastic_component_strain('XY')

        self.results = Results(
            nnum,
            nloc,
            nodal_stain_xx,
            nodal_stain_yy,
            nodal_stain_xy
        )

    @classmethod
    def case_creator_test(cls, resolution):
        fig, axs = plt.subplots(3, 3)
        fig.set_size_inches(10, 10)
        fig.set_dpi(500)

        for row in axs:
            for ax in row:
                case = cls.random(0, resolution)

                ax.add_patch(Rectangle((0, 0), case.width, case.height))
                ax.plot([case.left_support_start, case.left_support_start + case.left_support_length], [0, 0],
                        color='red')
                ax.plot([case.right_support_start, case.right_support_start + case.right_support_length], [0, 0],
                        color='red')
                ax.plot([case.force_start, case.force_start + case.force_length], [case.height, case.height],
                        color='green')
                ax.set(xlim=[-500, case.width + 500], ylim=[-500, case.height + 500])

        fig.show()
        fig.savefig('output/test.png', dpi=500)
