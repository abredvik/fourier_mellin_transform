import numpy as np

def unfold(img, ksize):
    if not (ksize % 2):
        raise ValueError("ksize must be an odd number")
    
    _, Y0 = np.meshgrid(np.arange(ksize ** 2), np.arange(img.shape[0]))
    _, Y1 = np.meshgrid(np.arange(ksize ** 2), np.arange(img.shape[1]))

    base0 = np.repeat(np.arange(ksize), ksize)
    base1 = np.tile(np.arange(ksize), ksize)

    indices0 = np.repeat(Y0 + base0, img.shape[1], axis=0)
    indices1 = np.tile(Y1 + base1, [img.shape[0], 1])
    return np.pad(img, ksize // 2)[indices0, indices1]

def median_filter(img, ksize=3):
    unfolded = unfold(img, ksize)
    medians = np.median(unfolded, axis=1)
    return medians.reshape(img.shape)

def blackman_window(shape):
    dim0_window = np.blackman(shape[0])
    dim1_window = np.blackman(shape[1])
    return np.outer(dim0_window, dim1_window)

def high_pass_filter(shape):
    dim0_window = np.blackman(shape[0])
    dim1_window = np.blackman(shape[1])
    result = np.outer(dim0_window, dim1_window)
    return np.abs(result - np.max(result))

def gaussian_kernel(ksize, sigma=1):
    if not (ksize % 2):
        raise ValueError("ksize must be an odd number")
    X, Y = np.meshgrid(*[np.linspace(-(ksize // 2), (ksize // 2), ksize)] * 2)
    distances = np.sqrt(X ** 2 + Y ** 2)
    norm = sigma * np.sqrt(2 * np.pi)
    result = np.exp(-np.square(distances) / (2 * sigma ** 2)) / norm
    return result / np.max(result)

def weighted_std_filter(spectrum):
    M = spectrum.shape[1]
    S = M / 4
    norm = np.exp(np.square(np.arange(M) - (M / 2)) / np.square(S))
    spectrum -= np.mean(spectrum, axis=1, keepdims=True)
    res = np.sqrt(np.sum(np.square(spectrum) / norm, axis=1))
    return np.repeat(res.reshape(-1, 1), M, axis=1)