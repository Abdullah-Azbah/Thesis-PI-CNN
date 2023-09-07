import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

width = 6000
height = 4000
x = np.arange(width)
y = np.arange(height)
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(2, 3)
fig.set(dpi=200)
fig.set_size_inches((10, 6))

material = np.ones((height, width)) * 30
material[200: 210, 200:width - 200] = 200
material[300: 310, 200:width - 200] = 200
material[400: 410, 200:width - 200] = 200
material[500: 510, 200:width - 200] = 200
material[1000: 1020, 200:width - 200] = 200
material[1500: 1520, 200:width - 200] = 200
material[2000: 2020, 200:width - 200] = 200
material[2500: 2520, 200:width - 200] = 200
material[3000: 3020, 200:width - 200] = 200
material[3500: 3520, 200:width - 200] = 200

material[200:height - 200, 750: 760] = 200
material[200:height - 200, 1500: 1510] = 200
material[200:height - 200, 2250: 2260] = 200
material[200:height - 200, 3000: 3010] = 200
material[200:height - 200, 3750: 3760] = 200
material[200:height - 200, 4500: 4510] = 200
material[200:height - 200, 5250: 5260] = 200

ax = axs[0, 0]
im = ax.contourf(X, Y, material)
ax.set(
    xlim=(-500, width + 500),
    ylim=(-500, height + 500),
    aspect='equal',
    title='Modulus of elasticity'
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)

support_x = np.ones((height, width))
support_x[0:50, 0:1000] = 0
support_x[0:50, 5000:] = 0

ax = axs[0, 1]
im = ax.contourf(X, Y, support_x)
ax.set(
    xlim=(-500, width + 500),
    ylim=(-500, height + 500),
    aspect='equal',
    title='Displacement X'
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)

support_y = np.ones((height, width))
support_y[0:50, 0:1000] = 0
support_y[0:50, 5000:] = 0

ax = axs[0, 2]
im = ax.contourf(X, Y, support_y)
ax.set(
    xlim=(-500, width + 500),
    ylim=(-500, height + 500),
    aspect='equal',
    title='Displacement Y'
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)

force_x = np.zeros((height, width))
force_x[0, 0] = 1

ax = axs[1, 0]
im = ax.contourf(X, Y, force_x)
ax.set(
    xlim=(-500, width + 500),
    ylim=(-500, height + 500),
    aspect='equal',
    title='Force X'
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)

force_x = np.zeros((height, width))
force_x[height - 50:height - 1, 2500:3500] = -1000

ax = axs[1, 1]
im = ax.contourf(X, Y, force_x)
ax.set(
    xlim=(-500, width + 500),
    ylim=(-500, height + 500),
    aspect='equal',
    title='Force Y'
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)

stress = pd.read_csv('proposal_figure_1_data.csv', delimiter='\t')
x = stress['X Location (mm)']
y = stress['Y Location (mm)']
stress = stress['Stress Intensity (MPa)']
stress[stress > 1e-3] = 1e-3
stress *= 10000
ax = axs[1, 2]

im = ax.tricontourf(x, y, stress)
ax.set(
    xlim=(-500, width + 500),
    ylim=(-500, height + 500),
    aspect='equal',
    title='Stress intensity'
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)

fig.tight_layout()
fig.show()
fig.savefig('proposal_figure_1.png', dpi=200)
