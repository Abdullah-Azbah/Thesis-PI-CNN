import numpy as np
import tensorflow as tf
from ThesisProject.util_functions import grad2d


def raw_physics_loss(input_tensor, output_tensor, dx=1, elasticity_layer_index=2):
    e_xx = output_tensor[..., 0]
    e_yy = output_tensor[..., 1]
    e_xy = output_tensor[..., 2]

    modulus_of_elasticity = input_tensor[..., elasticity_layer_index]

    # d_dx = grad2d(output_tensor, axis=2, dx=dx)
    # d_dy = grad2d(output_tensor, axis=1, dx=dx)
    #
    # du_dx = d_dx[..., 0]
    # dv_dx = d_dx[..., 1]
    # du_dy = d_dy[..., 0]
    # dv_dy = d_dy[..., 1]
    #
    # e_xx = du_dx
    # e_yy = dv_dy
    # e_xy = 0.5 * (du_dy + dv_dx)

    stress_xx = modulus_of_elasticity * (e_xx + 0.3 * e_yy)
    stress_yy = modulus_of_elasticity * (e_yy + 0.3 * e_xx)
    stress_xy = modulus_of_elasticity * (1 + 0.3) * e_xy

    stress_xx = stress_xx[..., tf.newaxis]
    stress_yy = stress_yy[..., tf.newaxis]
    stress_xy = stress_xy[..., tf.newaxis]

    d_stress_xx_x = grad2d(stress_xx, axis=2, dx=dx)
    d_stress_xy_y = grad2d(stress_xy, axis=1, dx=dx)
    d_stress_xy_x = grad2d(stress_xy, axis=2, dx=dx)
    d_stress_yy_y = grad2d(stress_yy, axis=1, dx=dx)

    res1 = d_stress_xx_x + d_stress_xy_y
    res2 = d_stress_xy_x + d_stress_yy_y

    return res1, res2


def mse_physics_loss(input_tensor, output_predict, dx=1, elasticity_layer_index=2, rate=1):
    res1, res2 = raw_physics_loss(
        input_tensor,
        output_predict,
        dx=dx,
        elasticity_layer_index=elasticity_layer_index
    )

    res1 = tf.reduce_mean(res1 ** 2)
    res2 = tf.reduce_mean(res2 ** 2)
    return rate * (res1 + res2)


def rmse_physics_loss(input_tensor, output_predict, dx=1, elasticity_layer_index=2, rate=1):
    r = mse_physics_loss(
        input_tensor, output_predict,
        dx=dx,
        elasticity_layer_index=elasticity_layer_index,
    )

    return rate * r ** 0.5


def mae_physics_loss(input_tensor, output_predict, dx=1, elasticity_layer_index=2, rate=1):
    res1, res2 = raw_physics_loss(
        input_tensor,
        output_predict,
        dx=dx,
        elasticity_layer_index=elasticity_layer_index
    )

    res1 = tf.reduce_mean(tf.abs(res1))
    res2 = tf.reduce_mean(tf.abs(res2))
    return rate * (res1 + res2)
