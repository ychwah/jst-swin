from numba import complex128, jit
import numpy as np


@jit(["complex128[:](complex128[:])", "complex128[:](float64[:])"], fastmath=True, cache=True, nopython=True)
def fft(input_array):
    """Computes the discrete Fourier transform of a sequence x using the iterative FFT algorithm.

        Args:
            input_array: A sequence of complex numbers of size 2N that should be a copy of the original

        Returns:
            The discrete Fourier transform of x.
    """
    N = len(input_array)
    m = N // 2
    n = int(np.log2(N))
    out_0 = np.zeros(input_array.shape, dtype=np.complex128)
    out_1 = np.zeros(input_array.shape, dtype=np.complex128)

    for r in range(1, n + 1):
        nb_blocks = 2 ** (n - r)
        curr_size = 2 ** r
        mid_size = 2 ** (r - 1)
        if r == 1:
            for i in range(nb_blocks):
                curr_block = curr_size * i
                for k in range(mid_size):
                    s = np.exp(-k * 1j * np.pi / mid_size)
                    out_0[curr_block + k] = input_array[i * mid_size + k] + s * input_array[i * mid_size + k + m]
                    out_0[curr_block + k + mid_size] = input_array[i * mid_size + k] - s * input_array[
                        i * mid_size + k + m]
        elif r % 2 == 0:
            for i in range(nb_blocks):
                curr_block = curr_size * i
                for k in range(mid_size):
                    s = np.exp(-k * 1j * np.pi / mid_size)
                    out_1[curr_block + k] = out_0[i * mid_size + k] + s * out_0[i * mid_size + k + m]
                    out_1[curr_block + k + mid_size] = out_0[i * mid_size + k] - s * out_0[i * mid_size + k + m]
        else:
            for i in range(nb_blocks):
                curr_block = curr_size * i
                for k in range(mid_size):
                    s = np.exp(-k * 1j * np.pi / mid_size)
                    out_0[curr_block + k] = out_1[i * mid_size + k] + s * out_1[i * mid_size + k + m]
                    out_0[curr_block + k + mid_size] = out_1[i * mid_size + k] - s * out_1[i * mid_size + k + m]
    if n % 2 == 0:
        return out_1
    else:
        return out_0


@jit(["complex128[:](complex128[:])", "complex128[:](float64[:])"], fastmath=True, cache=True, nopython=True)
def ifft(input_array):
    """Computes the inverse discrete Fourier transform of a sequence x.

    Args:
        input_array: A sequence of complex numbers.

    Returns:
        The inverse discrete Fourier transform of x.
    """

    N = len(input_array)
    input_array = input_array.conjugate()
    out = fft(input_array)
    out = out.conjugate()
    return out / N


@jit(["complex128[:, :](complex128[:, :])", "complex128[:, :](float64[:, :])"], fastmath=True, cache=True, nopython=True)
def fft2(input_array):
    """Computes the 2-dimensional discrete Fourier transform of an array x.

    Args:
        input_array: A 2-dimensional array of complex numbers.

    Returns:arallel=True,
        The 2-dimensional discrete Fourier transform of x.
    """

    N, M = input_array.shape
    out = np.zeros((N, M), dtype=np.complex128)
    for i in range(N):
        out[i, :] = fft(input_array[i, :])
    for j in range(M):
        out[:, j] = fft(out[:, j])
    return out


#
@jit(["complex128[:, :](complex128[:, :])", "complex128[:, :](float64[:, :])"], fastmath=True, cache=True, nopython=True)
def ifft2(input_array):
    """Computes the inverse 2-dimensional discrete Fourier transform of an array X.

    Args:
        input_array: A 2-dimensional array of complex numbers.

    Returns:
        The inverse 2-dimensional discrete Fourier transform of X.
    """

    N, M = input_array.shape
    x = np.zeros((N, M), dtype=complex128)
    for i in range(N):
        x[i, :] = ifft(input_array[i, :])
    for j in range(M):
        x[:, j] = ifft(x[:, j])
    return x / (N * M)