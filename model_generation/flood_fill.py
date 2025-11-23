from numba import jit, float64, int64, void
from numba.types import UniTuple


@jit(void(float64[:, :], UniTuple(int64, 2), float64), cache=True, nopython=True)
def flood_fill(image, seed_pixel, fill_color):
    """Fills the inside of a closed loop in an image with a given color.

  Args:
    image: A 2D numpy array representing the image.
    seed_pixel: A tuple of (row, column) coordinates representing the seed pixel from which to start filling.
    fill_color: The color to fill the inside of the closed loop with.

  Returns:
    A 2D numpy array representing the filled image.
  """

    # Create a queue to store the pixels to be filled.
    queue = [seed_pixel]

    # While the queue is not empty, fill the next pixel in the queue.
    while queue:
        row, column = queue.pop(0)

        # Check if the pixel is within the image bounds.
        if row < 0 or row >= image.shape[0] or column < 0 or column >= image.shape[1]:
            continue

        # Check if the pixel is already filled.
        if image[row, column] == fill_color:
            continue

        # Fill the pixel with the fill color.
        image[row, column] = fill_color

        # Add the neighboring pixels to the queue.
        for idx in [-1, 1]:
            if image[row + idx, column] != fill_color:
                queue.append((row + idx, column))
        for idy in [-1, 1]:
            if image[row, column + idy] != fill_color:
                queue.append((row, column + idy))
