import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from filters import blackman_window, high_pass_filter
from transformations import apply_transform, phase_correlation, warp_log_polar
from utils import compute_overlay, pad_to_shape, rgb2gray

def main(img0_path, img1_path):
    img0 = np.asarray(Image.open(img0_path), dtype='float32') / 255
    img1 = np.asarray(Image.open(img1_path), dtype='float32') / 255

    img0 = rgb2gray(img0)
    img1 = rgb2gray(img1)

    img0_windowed = img0 * blackman_window(img0.shape)
    img1_windowed = img1 * blackman_window(img1.shape)

    num_rows = max(img0.shape[0], img1.shape[0]) * 2
    num_cols = max(img0.shape[1], img1.shape[1]) * 2
    padded_shape = (num_rows, num_cols)

    img0_padded = pad_to_shape(img0_windowed, padded_shape)
    img1_padded = pad_to_shape(img1_windowed, padded_shape)

    img0_fft = np.fft.fftshift(np.fft.fft2(img0_padded))
    img1_fft = np.fft.fftshift(np.fft.fft2(img1_padded))

    high_pass = high_pass_filter(padded_shape)
    img0_fft = np.abs(img0_fft) * high_pass
    img1_fft = np.abs(img1_fft) * high_pass

    f,a=plt.subplots(1,2)
    a[0].imshow(img0_fft, cmap='gray')
    a[1].imshow(img1_fft, cmap='gray')
    plt.show()
    
    img0_logpolar = warp_log_polar(img0_fft)[:180]
    img1_logpolar = warp_log_polar(img1_fft)[:180]

    f,a=plt.subplots(1,2)
    a[0].imshow(img0_logpolar, cmap='gray')
    a[1].imshow(img1_logpolar, cmap='gray')
    plt.show()

    # determine rotation and scaling
    rs_correlation = phase_correlation(img0_logpolar, img1_logpolar)
    plt.imshow(rs_correlation)
    plt.show()

    coords = np.unravel_index(np.argmax(rs_correlation), rs_correlation.shape)
    ty, tx = coords - (np.array(rs_correlation.shape) // 2)

    y_center, x_center = padded_shape[0] / 2, padded_shape[1] / 2
    radius = np.sqrt(y_center ** 2 + x_center ** 2)
    scale = radius ** (tx / radius)
    angle = ty

    print('theta:', angle)
    print('scale:', scale)

    rs_transformed = apply_transform(img1, rotation=angle, scale=scale)

    plt.imshow(rs_transformed, cmap='gray')
    plt.show()

    trans_windowed = rs_transformed * blackman_window(rs_transformed.shape)
    trans_padded = pad_to_shape(trans_windowed, padded_shape)

    # determine translation
    tr_correlation = phase_correlation(img0_padded, trans_padded)
    plt.imshow(tr_correlation)
    plt.show()

    coords = np.unravel_index(np.argmax(tr_correlation), tr_correlation.shape)
    ty, tx = coords - (np.array(tr_correlation.shape) // 2)

    tr_transformed = apply_transform(rs_transformed, translation=(-tx, -ty), crop_center=True)

    fig, axes = plt.subplots(2, 2)
    axes[0,0].set_title('Image 0')
    axes[0,0].imshow(img0, cmap='gray')
    axes[0,1].set_title('Image 1')
    axes[0,1].imshow(img1, cmap='gray')
    axes[1,0].set_title('Image 1 Transformed')
    axes[1,0].imshow(tr_transformed, cmap='gray')
    axes[1,1].set_title('Overlay')
    axes[1,1].imshow(compute_overlay(img0, tr_transformed), cmap='gray')
    plt.show()

if __name__ == '__main__':
    # img0_path = 'EGaudi_1.jpg'
    # img1_path = 'EGaudi_2.jpg'
    img0_path = 'horse.png'
    img1_path = 'horse_rot_scale.png'
    # img0_path = 'im0_apodized.png'
    # img1_path = 'im1_apodized.png'
    main(img0_path, img1_path)
