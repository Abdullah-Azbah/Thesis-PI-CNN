import numpy as np

from ThesisProject.Cases import CaseV1
import matplotlib.pyplot as plt

case = CaseV1.from_file('output/cases_v1/000000.bin')

fields = case.create_matrices(1)

du_dx, du_dy = np.gradient(fields.displacement_x, 1, axis=[1, 0], edge_order=2)
dv_dx, dv_dy = np.gradient(fields.displacement_y, 1, axis=[1, 0], edge_order=2)

e_xx = du_dx
e_yy = dv_dy
e_xy = 0.5 * (du_dy + dv_dx)

stress_xx = case.modulus_of_elasticity * (e_xx + 0.3 * e_yy)
stress_yy = case.modulus_of_elasticity * (e_yy + 0.3 * e_xx)
stress_xy = case.modulus_of_elasticity * (1 + 0.3) * e_xy

stress_avg = 0.5 * (stress_xx + stress_yy)
R = np.sqrt((0.5 * (stress_xx - stress_yy)) ** 2 + stress_xy ** 2)

principal_min = stress_avg - R
principal_max = stress_avg + R


sigma = 0.5*np.max(np.stack([
    np.abs(principal_min),
    np.abs(principal_max),
    np.abs(principal_max - principal_min),
], -1), -1)

stress_intensity = np.abs(stress_xy)

fig, ax = plt.subplots(1, 1)
im = ax.contourf(sigma)
fig.colorbar(im)
ax.set(aspect='equal')
fig.show()

case.plot_result_component('stress_intensity')
