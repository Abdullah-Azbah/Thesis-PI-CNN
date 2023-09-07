import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ThesisProject.Blocks import res_block


class Model:
    pass


class ModelV1(Model):
    @staticmethod
    def get_model():
        inputs = tf.keras.layers.Input((26, 51, 3))

        model = tf.keras.layers.Conv2D(32, 9, padding='same')(inputs)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Activation('relu')(model)

        model = tf.keras.layers.Conv2D(64, 9, padding='same', strides=2)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Activation('relu')(model)

        model = tf.keras.layers.Conv2D(128, 9, padding='same', strides=2)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Activation('relu')(model)

        model = res_block(model, 128)
        model = res_block(model, 128)
        model = res_block(model, 128)
        model = res_block(model, 128)
        model = res_block(model, 128)

        model = res_block(model, 128)
        model = res_block(model, 128)
        model = res_block(model, 128)
        model = res_block(model, 128)
        model = res_block(model, 128)

        model = tf.keras.layers.Conv2DTranspose(64, 3, padding='same', strides=2)(model)
        model = tf.keras.layers.Activation('relu')(model)
        model = tf.keras.layers.BatchNormalization()(model)

        model = tf.keras.layers.Conv2DTranspose(32, 3, padding='same', strides=2)(model)
        model = tf.keras.layers.Activation('relu')(model)
        model = tf.keras.layers.BatchNormalization()(model)

        model = tf.keras.layers.Conv2D(1, 9, padding='same')(model)
        model = tf.keras.layers.Cropping2D(((1, 1), (1, 0)))(model)
        # model = tf.keras.layers.Activation('relu')(model)

        model = tf.keras.Model(inputs, model)
        return model

    @staticmethod
    def test_model(model, case):
        mae = tf.keras.losses.MeanAbsoluteError()

        input_matrix = case.get_input_matrix()[np.newaxis, :, :, :]
        output_matrix = case.get_output_matrix()

        model_output = model.predict(input_matrix)
        model_output = np.clip(model_output, -12, 5)

        print(mae(model_output[0, :, :, 0], output_matrix[:, :, 0]))

        fig, axs = plt.subplots(1, 2)
        im1 = axs[0].imshow(output_matrix[:, :, 0])
        im2 = axs[1].imshow(model_output[0, :, :, 0])

        fig.colorbar(im1)
        fig.colorbar(im2)
        fig.show()

    @staticmethod
    def save(model):
        output_path = os.path.join('output', 'ModelV1')
        model.save(output_path)