import numpy as np
from numba import jit, float64, void, int64
from flood_fill import flood_fill


@jit(float64[:, :](float64[:, :]), cache=True, nopython=True)
def split(points):
    n = points.shape[0]
    output = np.zeros((2 * n, 2))
    for idx in range(n):
        output[2 * idx] = points[idx]
        if idx == (n - 1):
            x, y = points[n - 1]
            x_next, y_next = points[0]
        else:
            x, y = points[idx]
            x_next, y_next = points[idx + 1]
        new_x = (x + x_next) / 2
        new_y = (y + y_next) / 2
        output[2 * idx + 1] = (new_x, new_y)
    return output


@jit(void(float64[:, :]), cache=True, nopython=True)
def average(points):
    n = len(points)
    for idx in range(n):
        if idx == (n - 1):
            x, y = points[n - 1]
            x_next, y_next = points[0]
        else:
            x, y = points[idx]
            x_next, y_next = points[idx + 1]

        new_x = (x + x_next) / 2
        new_y = (y + y_next) / 2
        points[idx] = (new_x, new_y)


@jit(float64[:, :](int64, int64), cache=True, nopython=True)
def create_shape(n, m):
    """Creates a random shape in a grayscale image format.

    Args:
    image: A grayscale image as a NumPy array.

    Returns:
    A grayscale image with a convex shape as a NumPy array.
    """

    # Initialize the output image.
    output_image = np.zeros((n, m))

    # Create a random center point for the convex shape.
    center_x = n // 2
    center_y = m // 2

    rad = min((n * 2) // 5, (m * 2) // 5)

    # Create a list of points on the circumference of the convex shape.
    nb_init_points = np.random.randint(10, 20)
    points = np.zeros((nb_init_points, 2))
    angles = [np.random.randint(0, 360) for _ in range(nb_init_points)]
    angles.sort()
    for idx, angle in enumerate(angles):
        radius = np.random.randint(10, rad)
        points[idx, 0] = center_x + radius * np.cos(angle * np.pi / 180)
        points[idx, 1] = center_y + radius * np.sin(angle * np.pi / 180)

    for idx in range(10):
        points = split(points)
        average(points)

    seed_point = (points[:, 0].mean(), points[:, 1].mean())

    for idx in range(points.shape[0]):
        p = points[idx]
        x, y = int(p[0]), int(p[1])
        output_image[x, y] = 1.0

    flood_fill(output_image, seed_point, 1.0)

    # check if fill was done succesfully. If it isn't the case, we invert the output: 0 <-> 1
    in_area = np.sum(output_image)
    out_area = n * m - in_area
    if in_area > out_area:
        output_image *= -1
        output_image += 1

    return output_image


if __name__ == "__main__":
    from misc import create_folder, save_img

    output_folder = "../data/test_structure"
    create_folder(output_folder)
    for i in range(100):
        structure = create_shape(64, 64)
        # for x in range(10, 50):
        #     for y in range(10, 50):
        #         if structure[x+1, y] > 0 and structure[x-1, y] > 0 and structure[x, y+1] > 0 and structure[x, y-1] > 0 :
        #             structure[x, y] = 1
        # print(f"{i} : {structure.sum()}")
        save_img(structure, f"{output_folder}/{i}.png")
