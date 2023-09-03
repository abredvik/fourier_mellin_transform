import numpy as np

def rgb2gray(img):
    if len(img.shape) == 2:
        return img
    elif len(img.shape) in [3, 4]:
        # 0.2125 R + 0.7154 G + 0.0721 B
        # Optimal weighting of Red, Green, and Blue according 
        # to how humans perceive color
        return np.dot(img[..., :3], [0.2125, 0.7154, 0.0721])
    raise ValueError("Input image is not grayscale or RGB")

def hann_window(shape):
    dim0_window = np.hamming(shape[0])
    dim1_window = np.hamming(shape[1])
    return np.outer(dim0_window, dim1_window)

def pad_to_shape(img, shape):
    if img.shape[0] > shape[0] or img.shape[1] > shape[1]:
        raise ValueError("Padded shape must be larger than image's shape")
    elif len(img.shape) != 2:
        raise ValueError("Image must be grayscale")
    
    axis0 = shape[0] - img.shape[0]
    axis1 = shape[1] - img.shape[1]
    pad_width = [
        (axis0 // 2, axis0 - axis0 // 2),
        (axis1 // 2, axis1 - axis1 // 2)
    ]

    return np.pad(img, pad_width)

def warp_log_polar(img):
    if len(img.shape) != 2:
        raise ValueError("Image must be grayscale")

    y_center, x_center = img.shape[0] / 2, img.shape[1] / 2
    radius = np.sqrt(y_center ** 2 + x_center ** 2)

    result = np.zeros((360, int(radius)))
    theta, rho = np.meshgrid(*[np.arange(d) for d in result.shape])

    # Rescale rho and theta before calculating x and y indices
    r = np.exp(rho * np.log(radius) / radius) - 1
    theta_rad = theta * np.pi / 180

    x_indices = np.int32(r * np.cos(theta_rad) + x_center).clip(0, img.shape[1] - 1)
    y_indices = np.int32(r * np.sin(theta_rad) + y_center).clip(0, img.shape[0] - 1)
    result[theta, rho] = img[y_indices, x_indices]

    return result

