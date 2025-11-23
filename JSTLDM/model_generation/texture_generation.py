import numpy as np
from numba import jit, float64, int64, boolean
from numba.types import UniTuple, string
from dfft import fft2, ifft2
from math import sqrt
from random import randint, random


@jit(float64[:, :](float64[:, :], int64), cache=True, nopython=True)
def P(input_array, r=5):
    step = r
    n, m = input_array.shape
    output = np.zeros((r ** 2, (n // step) * (m // step)))
    for i in range(n // step):
        for j in range(m // step):
            pos_x = i * step
            pos_y = j * step
            if pos_x + r > n:
                pos_x = n - r
            if pos_y + r > m:
                pos_y = m - r
            k = i * (m // step) + j
            for i_ in range(r):
                for j_ in range(r):
                    output[i_ * r + j_, k] = input_array[pos_x + i_, pos_y + j_]

    return output


@jit(float64[:, :](float64[:, :], int64, UniTuple(int64, 2)), cache=True, nopython=True)
def P_inv(input_array, r=5, output_shape=(50, 50)):
    output_array = np.zeros(output_shape)
    step = r
    n, m = output_shape
    for i in range(n // step):
        for j in range(m // step):
            pos_x = i * step
            pos_y = j * step
            if pos_x + r > n:
                pos_x = n - r
            if pos_y + r > m:
                pos_y = m - r
            k = i * (m // step) + j
            for i_ in range(r):
                for j_ in range(r):
                    output_array[pos_x + i_, pos_y + j_] = input_array[i_ * r + j_, k]

    return output_array


@jit(float64[:, :](int64, UniTuple(int64, 2), int64), cache=True, nopython=True)
def generate_uniform_texture_by_patch(rk, output_shape=(512, 512), patch_size=5):
    tmp_shape = (
        output_shape[0] if output_shape[0] % patch_size == 0 else (output_shape[0] // patch_size + 1) * patch_size,
        output_shape[1] if output_shape[1] % patch_size == 0 else (output_shape[1] // patch_size + 1) * patch_size
    )
    p_array_shape = P(np.ones(tmp_shape), r=patch_size).shape
    Y = np.zeros((p_array_shape[0], rk))
    X = np.zeros((p_array_shape[1], rk))
    for j in range(rk):
        for i in range(p_array_shape[0]):
            Y[i, j] = np.random.uniform(-1, 1)

        for i in range(p_array_shape[1]):
            X[i, j] = np.random.uniform(-1, 1)

    p_array = Y @ X.T
    output = P_inv(p_array, r=patch_size, output_shape=tmp_shape)
    output = output[:output_shape[0], :output_shape[1]]
    output -= output.mean()
    output *= 1.0 / max(abs(output.max()), abs(output.min()))

    return output


@jit(float64[:, :](int64, UniTuple(int64, 2), int64), cache=True, nopython=True)
def generate_gaussian_texture_by_patch(rk, output_shape=(512, 512), patch_size=5):
    tmp_shape = (
        output_shape[0] if output_shape[0] % patch_size == 0 else (output_shape[0] // patch_size + 1) * patch_size,
        output_shape[1] if output_shape[1] % patch_size == 0 else (output_shape[1] // patch_size + 1) * patch_size
    )
    p_array_shape = P(np.ones(tmp_shape), r=patch_size).shape
    Y = np.zeros((p_array_shape[0], rk))
    X = np.zeros((p_array_shape[1], rk))
    for j in range(rk):
        for i in range(p_array_shape[0]):
            Y[i, j] = np.random.normal()

        for i in range(p_array_shape[1]):
            X[i, j] = np.random.normal()

    p_array = Y @ X.T
    output = P_inv(p_array, r=patch_size, output_shape=tmp_shape)
    output = output[:output_shape[0], :output_shape[1]]
    output -= output.mean()
    output *= 1.0 / max(abs(output.max()), abs(output.min()))

    return output


@jit(float64[:, :](int64, UniTuple(int64, 2)), cache=True, nopython=True)
def generate_fourier_texture(rk, output_shape=(512, 512)):
    # first create a bigger array that can be divided by the patchsize
    array_hat = np.zeros(output_shape, dtype=np.complex128)
    output_shape = np.array(output_shape)
    center = (output_shape / 2) - 0.5
    indexes = np.zeros((rk, 2), dtype=np.int64)
    indexes[:, 0] = np.random.randint(0, output_shape[0], rk)
    indexes[:, 1] = np.random.randint(0, output_shape[1], rk)

    for index in indexes:

        centered_index = 2 * (index - center) / output_shape
        centered_l2_norm = np.sum(np.abs(centered_index)**2)
        if centered_l2_norm > 1.0:
            tmp = index - (centered_l2_norm - 1) * np.sign(centered_index) * center
            index = np.array([int(tmp[0]), int(tmp[1])], dtype=int64)
        # centered_norm = sqrt(np.sum(centered_index ** 2))
        amp = np.random.uniform(0.8, 1.2)
        array_hat[index[0], index[1]] = amp
        array_hat[-index[0], -index[1]] = amp

    output = ifft2(array_hat).real
    output *= 1.0 / max(abs(output.max()), abs(output.min()))

    return output


@jit(float64[:, :](int64, UniTuple(int64, 2), int64), cache=True, nopython=True)
def generate_fourier_texture_by_patch(rk, output_shape=(512, 512), patch_size=5):
    # first create a bigger array that can be divided by the patchsize
    tmp_shape = (
        output_shape[0] if output_shape[0] % patch_size == 0 else (output_shape[0] // patch_size + 1) * patch_size,
        output_shape[1] if output_shape[1] % patch_size == 0 else (output_shape[1] // patch_size + 1) * patch_size)

    array_hat = generate_fourier_texture(rk, output_shape=tmp_shape)
    p_fourier = P(array_hat, r=patch_size)
    u, d, v = np.linalg.svd(p_fourier, full_matrices=False)
    d[rk:] = 0
    p_fourier = u @ np.diag(d) @ v
    output = P_inv(p_fourier, r=patch_size, output_shape=tmp_shape)
    output = output[:output_shape[0], :output_shape[1]]
    output *= 1.0 / max(abs(output.max()), abs(output.min()))

    return output


@jit(float64[:, :](UniTuple(int64, 2)), cache=True, nopython=True)
def generate_random_texture(output_shape):
    # random parameters
    rk = randint(1, 4)
    texture_type = random()
    if texture_type < 0.4:
        texture = generate_fourier_texture(rk, output_shape)
    elif texture_type < 0.8:
        texture = generate_uniform_texture_by_patch(rk, output_shape, 4)
    else:
        texture = np.zeros(output_shape)

    return texture


if __name__ == "__main__":
    from misc import save_img, create_folder

    create_folder("../tmp_uniform")

    for i in range(10):
        toto = 0.2 * generate_uniform_texture_by_patch(rk=2, output_shape=(128, 128), patch_size=4)
        # toto = 0.2 * generate_random_texture(output_shape=(64, 64))
        save_img(toto + 0.5, f"../tmp_uniform/{i}.png")
