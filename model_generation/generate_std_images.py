import numpy as np
from numba import jit, float64, int64
from numba.types import UniTuple
from shape_generation import create_shape
from texture_generation import generate_random_texture
from misc import save_img


@jit(float64[:, :](float64[:, :]), cache=True, nopython=True)
def shear_image(input_array):
    """
    This cuts away the unnecessary columns and rows with only zeros in the input array
    Note that we also fix holes in the image.
    :param input_array: image with many zeros
    :return: a sheared version of the image
    """
    n, m = input_array.shape
    bottom_cut_point_x = n - 1
    top_cut_point_x = 0
    bottom_cut_point_y = m - 1
    top_cut_point_y = 0
    for idx in range(1, n - 1):
        for idy in range(1, m - 1):
            x = input_array[idx, idy]
            # hole fixing
            if x > 0:
                if idx < bottom_cut_point_x:
                    bottom_cut_point_x = idx
                if idy < bottom_cut_point_y:
                    bottom_cut_point_y = idy
                if idx > top_cut_point_x:
                    top_cut_point_x = idx
                if idy > top_cut_point_y:
                    top_cut_point_y = idy
            else:
                if input_array[idx + 1, idy] + input_array[idx - 1, idy] + input_array[idx, idy + 1] + input_array[idx, idy - 1] == 4.0:
                    input_array[idx, idy] = 1.0
    output = input_array[bottom_cut_point_x:top_cut_point_x, bottom_cut_point_y:top_cut_point_y]
    return output


@jit(UniTuple(float64[:, :], 2)(int64, int64), cache=True, nopython=True)
def generate_std(n=512, m=512):
    # random parameters
    # nb_of_convex_shapes = np.random.randint(10, 20)
    # nb_of_convex_shapes = np.random.randint(1, 6)
    nb_of_convex_shapes = np.random.randint(1, 3)
    texture = generate_random_texture((n, m))  # background texture
    cartoon_amp = np.random.uniform(0.2, 0.8)
    texture_amp = np.random.uniform(0.01, 0.2)
    texture *= texture_amp / np.max(np.abs(texture))
    structure = np.ones((n, m)) * cartoon_amp
    for _ in range(nb_of_convex_shapes):
        input_cartoon = create_shape(n, m)

        input_cartoon = shear_image(input_cartoon)
        n_, m_ = input_cartoon.shape
        input_texture = generate_random_texture((n, m))

        cartoon_amp = np.random.uniform(0.2, 0.8)
        texture_amp = np.random.uniform(0.01, min(1.0 - cartoon_amp, 0.2))
        input_texture *= (texture_amp / np.max(np.abs(input_texture)))

        center_x, center_y = (n_ // 2, m_ // 2)
        pos_x, pos_y = (np.random.randint(0, n), np.random.randint(0, m))
        start_x, start_y = max(pos_x - center_x, 0), max(pos_y - center_y, 0)
        end_x, end_y = min(pos_x + n_ - center_x, n), min(pos_y + m_ - center_y, m)
        for i, idx in enumerate(range(start_x, end_x, 1)):
            for j, idy in enumerate(range(start_y, end_y, 1)):
                x = center_x - (pos_x - start_x) + i
                y = center_y - (pos_y - start_y) + j
                if input_cartoon[x, y] > 0:
                    structure[idx, idy] = cartoon_amp
                    texture[idx, idy] = input_texture[idx, idy]

    return structure, texture


@jit(float64[:, :](int64, int64), cache=True, nopython=True)
def generate_structure(n=512, m=512):
    # random parameters
    nb_of_convex_shapes = np.random.randint(2, 12)
    structure = np.ones((n, m)) * np.random.uniform(0.2, 0.8)
    for _ in range(nb_of_convex_shapes):
        input_cartoon = create_shape(n, m)

        input_cartoon = shear_image(input_cartoon)
        n_, m_ = input_cartoon.shape
        cartoon_amp = np.random.uniform(0.2, 0.8)
        center_x, center_y = (n_ // 2, m_ // 2)
        pos_x, pos_y = (np.random.randint(0, n), np.random.randint(0, m))
        start_x, start_y = max(pos_x - center_x, 0), max(pos_y - center_y, 0)
        end_x, end_y = min(pos_x + n_ - center_x, n), min(pos_y + m_ - center_y, m)
        for i, idx in enumerate(range(start_x, end_x, 1)):
            for j, idy in enumerate(range(start_y, end_y, 1)):
                x = center_x - (pos_x - start_x) + i
                y = center_y - (pos_y - start_y) + j
                if input_cartoon[x, y] > 0:
                    structure[idx, idy] = cartoon_amp

    return structure


@jit(float64[:, :](int64, int64), cache=True, nopython=True)
def generate_texture(n=512, m=512):
    # random parameters
    texture = generate_random_texture((n, m))
    texture_amp = np.random.uniform(0.05, 0.25)
    texture *= texture_amp / np.max(np.abs(texture))

    return texture


if __name__ == "__main__":
    from misc import create_folder
    output_folder = "../data/tmp_denoise"
    create_folder(f"{output_folder}")
    for iter_ in range(100):
        u, v = generate_std(256, 256)
        save_img(u, f"{output_folder}/structure_{iter_}.png")
        save_img(v + 0.5, f"{output_folder}/texture_{iter_}.png")
        save_img(u + v, f"{output_folder}/image_{iter_}.png")

        save_img(u + np.random.normal(0, 0.03, size=(256, 256)), f"{output_folder}/noisy_structure_{iter_}.png")
        save_img(v + np.random.normal(0, 0.03, size=(256, 256)) + 0.5, f"{output_folder}/noisy_texture_{iter_}.png")