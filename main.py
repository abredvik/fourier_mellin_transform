import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from skimage.transform import warp_polar

from utils import rgb2gray, hann_window, pad_to_shape, warp_log_polar

def main(img0_path, img1_path):
    img0 = np.asarray(Image.open(img0_path), dtype='float32') / 255
    img1 = np.asarray(Image.open(img1_path), dtype='float32') / 255

    img0 = rgb2gray(img0)
    img1 = rgb2gray(img1)

    res = warp_log_polar(img0)
    f,a=plt.subplots(1,2)
    a[0].imshow(res, cmap='gray')
    a[1].imshow(warp_polar(img0, scaling='log'), cmap='gray')
    plt.show()

    # img0_windowed = img0 * hann_window(img0.shape)
    # img1_windowed = img1 * hann_window(img1.shape)

    # num_rows = max(img0.shape[0], img1.shape[0]) * 2
    # num_cols = max(img0.shape[1], img1.shape[1]) * 2
    # padded_shape = (num_rows, num_cols)

    # img0_padded = pad_to_shape(img0_windowed, padded_shape)
    # img1_padded = pad_to_shape(img1_windowed, padded_shape)

    # img0_fft = np.fft.fftshift(np.fft.rfft2(img0_padded), axes=0)
    # img1_fft = np.fft.fftshift(np.fft.rfft2(img1_padded), axes=0)
    


    




if __name__ == '__main__':
    img0_path = 'horse.png'
    # img0_path = 'Danny-DiVito-as-the-Trashman.jpg'
    img1_path = 'horse_rot_scale.png'
    # img1_path = 'horse.png'
    main(img0_path, img1_path)
