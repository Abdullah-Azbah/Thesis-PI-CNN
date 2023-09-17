import numpy as np
import tensorflow as tf
from ThesisProject.util_functions import grad2d


def physics_raw_loss(input_tensor, output_tensor, dx=1, elasticity_layer_index=2):
    # e_xx = output_tensor[..., 0]
    # e_yy = output_tensor[..., 1]
    # e_xy = output_tensor[..., 2]

    modulus_of_elasticity = input_tensor[..., elasticity_layer_index]

    d_dx = grad2d(output_tensor, axis=2, dx=dx)
    d_dy = grad2d(output_tensor, axis=1, dx=dx)

    du_dx = d_dx[..., 0]
    dv_dx = d_dx[..., 1]
    du_dy = d_dy[..., 0]
    dv_dy = d_dy[..., 1]

    e_xx = du_dx
    e_yy = dv_dy
    e_xy = du_dy + dv_dx

    stress_xx = modulus_of_elasticity * (e_xx + 0.3 * e_yy)
    stress_yy = modulus_of_elasticity * (e_yy + 0.3 * e_xx)
    stress_xy = 0.5 * modulus_of_elasticity * (1 + 0.3) * e_xy

    stress_xx = stress_xx[..., tf.newaxis]
    stress_yy = stress_yy[..., tf.newaxis]
    stress_xy = stress_xy[..., tf.newaxis]

    d_stress_xx_x = grad2d(stress_xx, axis=2, dx=dx)
    d_stress_xy_y = grad2d(stress_xy, axis=1, dx=dx)
    d_stress_xy_x = grad2d(stress_xy, axis=2, dx=dx)
    d_stress_yy_y = grad2d(stress_yy, axis=1, dx=dx)

    d_exx_y = grad2d(e_xx[..., np.newaxis], axis=1, dx=dx)
    d_eyy_x = grad2d(e_yy[..., np.newaxis], axis=2, dx=dx)

    d2_exx_y = grad2d(d_exx_y, axis=1, dx=dx)
    d2_eyy_x = grad2d(d_eyy_x, axis=2, dx=dx)

    d_exy_x = grad2d(e_xy[..., np.newaxis], axis=2, dx=dx)
    d2_exy_xy = grad2d(d_exy_x, axis=1, dx=dx)

    eq_x = d_stress_xx_x + d_stress_xy_y
    eq_y = d_stress_xy_x + d_stress_yy_y
    comb = d2_exx_y + d2_eyy_x - d2_exy_xy

    return eq_x, eq_y, comb


def physics_mse_loss(input_tensor, output_predict, dx=1, elasticity_layer_index=2, rate=1):
    if isinstance(rate, (int, float)):
        rate = [rate] * 3
    elif not isinstance(rate, list) or (isinstance(rate, list) and len(rate) != 3):
        raise ValueError('rate must be float or a list of 3 floats')

    eq_x, eq_y, comb = physics_raw_loss(
        input_tensor,
        output_predict,
        dx=dx,
        elasticity_layer_index=elasticity_layer_index
    )

    eq_x = tf.reduce_mean(eq_x ** 2)
    eq_y = tf.reduce_mean(eq_y ** 2)
    comb = tf.reduce_mean(comb ** 2)

    return rate[0] * eq_x + rate[1] * eq_y + rate[2] * comb


def physics_rmse_loss(input_tensor, output_predict, dx=1, elasticity_layer_index=2, rate=1):
    if isinstance(rate, (int, float)):
        rate = [rate] * 3
    elif not isinstance(rate, list) or (isinstance(rate, list) and len(rate) != 3):
        raise ValueError('rate must be float or a list of 3 floats')

    eq_x, eq_y, comb = physics_raw_loss(
        input_tensor, output_predict,
        dx=dx,
        elasticity_layer_index=elasticity_layer_index,
    )

    eq_x = tf.reduce_mean(eq_x ** 2) ** 0.5
    eq_y = tf.reduce_mean(eq_y ** 2) ** 0.5
    comb = tf.reduce_mean(comb ** 2) ** 0.5

    return rate[0] * eq_x + rate[1] * eq_y + rate[2] * comb


def physics_mae_loss(input_tensor, output_predict, dx=1, elasticity_layer_index=2, rate=1):
    if isinstance(rate, (int, float)):
        rate = [rate] * 3
    elif not isinstance(rate, list) or (isinstance(rate, list) and len(rate) != 3):
        raise ValueError('rate must be float or a list of 3 floats')

    eq_x, eq_y, comb = physics_raw_loss(
        input_tensor,
        output_predict,
        dx=dx,
        elasticity_layer_index=elasticity_layer_index
    )

    eq_x = tf.reduce_mean(tf.abs(eq_x))
    eq_y = tf.reduce_mean(tf.abs(eq_y))
    comb = tf.reduce_mean(tf.abs(comb))

    return rate[0] * eq_x + rate[1] * eq_y + rate[2] * comb
