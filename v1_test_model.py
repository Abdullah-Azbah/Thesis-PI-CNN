import os
import random

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import tensorflow as tf

from ThesisProject.FieldMatrices import FieldMatrices
from ThesisProject.Cases import CaseV1

N = 3
RESOLUTION = 100
MATRICES_OUTPUT_DIR = f'output/cases_v1_res{RESOLUTION}'


def main():
    files = os.listdir(MATRICES_OUTPUT_DIR)
    files = [random.choice(files) for _ in range(N)]

    input_matrices = []
    output_matrices = []
    for file in files:
        case = FieldMatrices.from_file(os.path.join(MATRICES_OUTPUT_DIR, file))
        input_matrices.append(case.get_input_matrix())
        output_matrices.append(case.get_output_matrix())

    input_matrices = np.stack(input_matrices)

    model = tf.keras.models.load_model('output/ModelV1_physics_mse', custom_objects={'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError})
    predict = model.predict(input_matrices)

    fig, axs = plt.subplots(2, N)
    fig.set_figwidth(11)
    fig.set_figheight(3)
    fig.set_dpi(200)

    for i in range(N):
        p = predict[i]
        r = output_matrices[i]

        new_size = np.array(p.shape[1::-1]) * RESOLUTION
        p = cv2.resize(p, new_size)
        r = cv2.resize(r, new_size)

        p = axs[0, i].imshow(p[..., 0], origin='lower')
        r = axs[1, i].imshow(r[..., 0], origin='lower')

        divider = make_axes_locatable(axs[0, i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(p, cax=cax)

        divider = make_axes_locatable(axs[1, i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(r, cax=cax)

    fig.subplots_adjust(
        wspace=0.50,
        hspace=0,
        top=1,
        bottom=0.05,
        left=0.1,
        right=0.925
    )
    # fig.tight_layout()
    fig.show()
    # fig.savefig('proposal_figure_2.png', dpi=200)


if __name__ == '__main__':
    main()
