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
        inputs = tf.keras.layers.Input((None, None, 3))

        outputs = tf.keras.layers.Conv2D(32, 9, padding='same')(inputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)

        outputs = tf.keras.layers.Conv2D(64, 9, padding='same', strides=2)(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)

        outputs = tf.keras.layers.Conv2D(128, 9, padding='same', strides=2)(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)

        outputs = res_block(outputs, 128)
        outputs = res_block(outputs, 128)
        outputs = res_block(outputs, 128)
        outputs = res_block(outputs, 128)
        outputs = res_block(outputs, 128)

        outputs = res_block(outputs, 128)
        outputs = res_block(outputs, 128)
        outputs = res_block(outputs, 128)
        outputs = res_block(outputs, 128)
        outputs = res_block(outputs, 128)

        outputs = tf.keras.layers.Conv2DTranspose(64, 3, padding='same', strides=2)(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)

        outputs = tf.keras.layers.Conv2DTranspose(32, 3, padding='same', strides=2)(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)

        outputs = tf.keras.layers.Conv2D(2, 9, padding='same')(outputs)
        outputs = tf.keras.layers.Cropping2D(((1, 1), (1, 0)))(outputs)
        # model = tf.keras.layers.Activation('relu')(model)

        model = tf.keras.Model(inputs, outputs)

        return inputs, outputs, model

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
    def save(model, suffix=''):
        output_path = os.path.join('output', 'ModelV1' + '_' + suffix)
        model.save(output_path)
