import tensorflow as tf


def se_block(inputs, filters):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    squeeze = tf.keras.layers.Dense(filters // 16, activation='relu')(squeeze)
    excitation = tf.keras.layers.Dense(filters, activation='sigmoid')(squeeze)
    excitation = tf.keras.layers.Reshape((1, 1, filters))(excitation)

    scaled = tf.keras.layers.Multiply()([inputs, excitation])

    return scaled


def res_block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = se_block(x, filters)

    x = tf.keras.layers.Add()([inputs, x])

    return x
