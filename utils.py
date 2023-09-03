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

def pad_to_shape(img, shape):
    if img.shape[0] > shape[0] or img.shape[1] > shape[1]:
        raise ValueError("Padded shape must be larger than image's shape")
    elif len(img.shape) != 2:
        raise ValueError("Image must be grayscale")
    
    axis0 = shape[0] - img.shape[0]
    axis1 = shape[1] - img.shape[1]
    pad_width = [
        (axis0 // 2, axis0 - (axis0 // 2)),
        (axis1 // 2, axis1 - (axis1 // 2))
    ]

    return np.pad(img, pad_width)

def compute_overlay(img0, img1):
    num_rows = max(img0.shape[0], img1.shape[0])
    num_cols = max(img0.shape[1], img1.shape[1])

    img0_ax0 = max(0, num_rows - img0.shape[0])
    img1_ax0 = max(0, num_rows - img1.shape[0])
    img0_ax1 = max(0, num_cols - img0.shape[1])
    img1_ax1 = max(0, num_cols - img1.shape[1])

    pad_width0 = [
        (img0_ax0 // 2, img0_ax0 - (img0_ax0 // 2)),
        (img0_ax1 // 2, img0_ax1 - (img0_ax1 // 2))
    ]
    pad_width1 = [
        (img1_ax0 // 2, img1_ax0 - (img1_ax0 // 2)),
        (img1_ax1 // 2, img1_ax1 - (img1_ax1 // 2))
    ]

    img0_padded = np.pad(img0, pad_width0)
    img1_padded = np.pad(img1, pad_width1)

    return (img0_padded + img1_padded) / 2