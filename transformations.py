import numpy as np

from utils import pad_to_shape
from filters import gaussian_kernel, median_filter

def phase_correlation(f, g, blur=False):
    F_f = np.fft.fft2(f)
    F_g = np.fft.fft2(g)
    R = F_f * np.conj(F_g)
    R /= np.abs(R)
    if blur:
        gauss = np.fft.fft2(pad_to_shape(gaussian_kernel(5), R.shape))
        return np.fft.ifft2(R * gauss).real
    else:
        r = np.fft.ifft2(R).real
        return np.fft.ifftshift(r)

def warp_log_polar(img):
    if len(img.shape) != 2:
        raise ValueError("Image must be grayscale")

    y_center, x_center = img.shape[0] / 2, img.shape[1] / 2
    radius = np.sqrt(y_center ** 2 + x_center ** 2)

    result = np.zeros((360, int(radius)))
    theta, r = np.meshgrid(*[np.arange(d) for d in result.shape])

    rho = radius ** (r / radius)
    theta_rad = theta * np.pi / 180

    y_indices = np.int32(rho * np.sin(theta_rad) + y_center).clip(0, img.shape[0] - 1)
    x_indices = np.int32(rho * np.cos(theta_rad) + x_center).clip(0, img.shape[1] - 1)
    result[theta, r] = img[y_indices, x_indices]

    return result

def apply_transform(img, translation=(0,0), rotation=0, scale=1, crop_center=False):
    s, theta, (ty, tx) = scale, rotation * np.pi / 180, translation
    M = np.array([[s * np.cos(theta), -s * np.sin(theta), tx],
                  [s * np.sin(theta),  s * np.cos(theta), ty],
                  [                0,                  0,  1]])
    M_inv = np.linalg.pinv(M)

    result = np.zeros(3 * np.array(img.shape))
    old_coords = np.vstack(np.where(np.ones_like(img)) + (np.ones((img.size), dtype='int32'),))

    new_coords = np.int32(M_inv @ old_coords + np.array(img.shape + (0,)).reshape(-1, 1))
    result[new_coords[0], new_coords[1]] = img[old_coords[0], old_coords[1]]

    if crop_center:
        ymin, ymax = img.shape[0], 2 * img.shape[0]
        xmin, xmax = img.shape[1], 2 * img.shape[1]
    else:
        ymin, ymax = np.min(new_coords[0]), np.max(new_coords[0]) + 1
        xmin, xmax = np.min(new_coords[1]), np.max(new_coords[1]) + 1

    return median_filter(result[ymin:ymax,xmin:xmax])
