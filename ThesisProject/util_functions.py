import os
import pickle
import random
import tensorflow as tf


def get_random_token(n=6):
    rng = random.Random()
    lower = 16 ** n
    upper = 16 ** (n + 1) - 1

    token = hex(rng.randint(lower, upper))[2:]

    return token


def create_case_generator_from_directory(directory, cls, epochs=1, batch_size=0):
    for _ in range(epochs):
        for file in os.listdir(directory):
            yield cls.from_file(os.path.join(directory, file))


def round_nearest(x, base):
    return base * round(x / base)


def grad2d(tensor, axis, dx=1.0):
    if axis == 1:
        d_center = (tensor[:, 2:, :, :] - tensor[:, :-2, :, :]) / (2.0 * dx)
        d_edge1 = (-1.5 * tensor[:, 0, :, :] + 2.0 * tensor[:, 1, :, :] - 0.5 * tensor[:, 2, :, :]) / dx
        d_edge2 = (1.5 * tensor[:, -1, :, :] - 2.0 * tensor[:, -2, :, :] + 0.5 * tensor[:, -3, :, :]) / dx
        d = tf.concat([d_edge1[:, tf.newaxis, :, :], d_center, d_edge2[:, tf.newaxis, :, :]], axis=1)
    elif axis == 2:
        d_center = (tensor[:, :, 2:, :] - tensor[:, :, :-2, :]) / (2.0 * dx)
        d_edge1 = (-1.5 * tensor[:, :, 0, :] + 2.0 * tensor[:, :, 1, :] - 0.5 * tensor[:, :, 2, :]) / dx
        d_edge2 = (1.5 * tensor[:, :, -1, :] - 2.0 * tensor[:, :, -2, :] + 0.5 * tensor[:, :, -3, :]) / dx
        d = tf.concat([d_edge1[:, :, tf.newaxis, :], d_center, d_edge2[:, :, tf.newaxis, :]], axis=2)
    else:
        raise ValueError("Invalid axis; must be 2 (x-axis) or 1 (y-axis)")

    return d
