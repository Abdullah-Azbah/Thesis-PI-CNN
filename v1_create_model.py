import numpy as np
import tensorflow as tf
import os
import random

from ThesisProject.Cases import CaseV1
from ThesisProject.Models import ModelV1
from ThesisProject.util_functions import create_case_generator_from_directory
from ThesisProject.loses import physics_mse_loss

N = 10_000
RESOLUTION = 100
CASE_OUTPUT_DIR = f'output/cases_v1_res{RESOLUTION}'
EPOCHS = 5
RES_BLOCK_FILTERS = 128
BATCH_SIZE = 20
STEPS_PER_EPOCH = N // BATCH_SIZE


def generator(directory, epochs=1, batch_size=1, as_numpy_array=False):
    for epoch in range(epochs):
        files = os.listdir(directory)
        while files:
            slice, files = files[0:batch_size], files[batch_size:]
            input_data = []
            output_data = []
            for file in slice:
                case = CaseV1.from_file(os.path.join(directory, file))

                input_matrix = case.get_input_matrix()
                output_matrix = case.get_output_matrix()

                input_matrix = input_matrix[np.newaxis, :, :, :]
                output_matrix = output_matrix[np.newaxis, :, :, :]

                input_data.append(tf.constant(input_matrix))
                output_data.append(tf.constant(output_matrix))

            if as_numpy_array:
                i = tf.concat(input_data, 0)
                o = tf.concat(output_data, 0)

                yield i, o
            else:
                data = list(zip(input_data, output_data))
                for i, o in data:
                    yield i, o


def generator_fake(directory, epochs=1, batch_size=1):
    file = os.listdir(directory)[0]

    case = CaseV1.from_file(os.path.join(directory, file))

    input_matrix = case.get_input_matrix()[np.newaxis, :, :, :]
    output_matrix = case.get_output_matrix()[np.newaxis, :, :, :]

    for epoch in range(epochs):
        for _ in range(1000):
            yield input_matrix, output_matrix


def main():
    data = generator(CASE_OUTPUT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, as_numpy_array=True)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch, rate: 0.001 * 0.1 ** epoch)

    inputs, outputs, model = ModelV1.get_model()
    model.summary()

    physics_loss = physics_mse_loss(inputs, outputs, dx=RESOLUTION, rate=[0, 0, 10e24])
    model.add_loss(physics_loss)
    model.add_metric(physics_loss, 'physics')

    model.compile(loss='mae', optimizer='adam')
    model.fit(data, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_scheduler])

    ModelV1.save(model, suffix='physics_mse')


if __name__ == '__main__':
    main()
