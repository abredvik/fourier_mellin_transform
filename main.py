import numpy as np
from PIL import Image

from filters import blackman_window, high_pass_filter, weighted_std_filter
from transformations import apply_transform, phase_correlation, warp_log_polar
from utils import compute_overlay, pad_to_shape, rgb2gray

def main(img0_path, img1_path):
    # Read images with PIL
    img0 = np.asarray(Image.open(img0_path), dtype='float32') / 255
    img1 = np.asarray(Image.open(img1_path), dtype='float32') / 255

    # Convert to grayscale
    img0 = rgb2gray(img0)
    img1 = rgb2gray(img1)

    # Apply blackman window
    img0_windowed = img0 * blackman_window(img0.shape)
    img1_windowed = img1 * blackman_window(img1.shape)

    # Determine a common padded shape
    num_rows = max(img0.shape[0], img1.shape[0]) * 2
    num_cols = max(img0.shape[1], img1.shape[1]) * 2
    padded_shape = (num_rows, num_cols)

    # Pad each image
    img0_padded = pad_to_shape(img0_windowed, padded_shape)
    img1_padded = pad_to_shape(img1_windowed, padded_shape)

    # Compute fft
    img0_fft = np.fft.fftshift(np.fft.fft2(img0_padded))
    img1_fft = np.fft.fftshift(np.fft.fft2(img1_padded))

    # Apply a high pass filter
    high_pass = high_pass_filter(padded_shape)
    img0_fft = np.abs(img0_fft) * high_pass
    img1_fft = np.abs(img1_fft) * high_pass
    
    # Warp to log-polar coordinates
    img0_logpolar = warp_log_polar(img0_fft)[:180]
    img1_logpolar = warp_log_polar(img1_fft)[:180]

    # Apply weighted standard deviaton filter
    img0_logpolar *= weighted_std_filter(img0_logpolar)
    img1_logpolar *= weighted_std_filter(img1_logpolar)

    # Compute phase correlation (for rotation + scale)
    rs_correlation = phase_correlation(img0_logpolar, img1_logpolar)
    coords = np.unravel_index(np.argmax(rs_correlation), rs_correlation.shape)
    ty, tx = coords - (np.array(rs_correlation.shape) // 2)

    # Determine rotation and scale
    y_center, x_center = padded_shape[0] / 2, padded_shape[1] / 2
    radius = np.sqrt(y_center ** 2 + x_center ** 2)
    scale = round(radius ** (tx / radius), 1)
    angle = ty

    print('theta:', angle)
    print('scale:', scale)

    # Apply the rotation and scaling to the image
    rs_transformed = apply_transform(img1, rotation=angle, scale=scale)
    trans_windowed = rs_transformed * blackman_window(rs_transformed.shape)
    trans_padded = pad_to_shape(trans_windowed, padded_shape)

    # Compute phase correlation (for translation)
    tr_correlation = phase_correlation(img0_padded, trans_padded, blur=True)
    coords = np.unravel_index(np.argmax(tr_correlation), tr_correlation.shape)
    ty, tx = coords - (np.array(tr_correlation.shape) // 2)

    # Apply the translation
    tr_transformed = apply_transform(rs_transformed, translation=(-tx, -ty), crop_center=True)

    # Compute an overlay
    overlay = compute_overlay(img0, tr_transformed)

    # Save the results with PIL
    Image.fromarray(np.uint8(tr_transformed * 255)).save('transformed_result.png')
    Image.fromarray(np.uint8(overlay * 255)).save('overlayed_result.png')

if __name__ == '__main__':
    img0_path = 'demo/horse.png'
    img1_path = 'demo/horse_rot_scale.png'
    main(img0_path, img1_path)
