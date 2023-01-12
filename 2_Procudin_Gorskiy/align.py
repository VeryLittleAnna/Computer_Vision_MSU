import numpy as np
import matplotlib as plt
from skimage.io import imread, imsave, imshow
from matplotlib import pyplot as plt
from copy import copy

def crop(source_img): #bgr
    n, m = source_img.shape; n //= 3
    dn = int(n * 0.07); dm = int(m * 0.07)
    blue = source_img[dn:n-dn, dm:m-dm]
    green = source_img[n+dn:2*n-dn, dm:m-dm]
    red = source_img[2*n+dn:3*n-dn, dm:m-dm]
    
    canals = np.dstack([blue, green, red]) 
    # plt.imshow(canals)
    return canals, n, m


def find_shift(img1, img2):
    """
    img1, img2 - shape (n, m)
    """
    C = np.fft.ifft2(np.fft.fft2(img1) * np.conj(np.fft.fft2(img2)))
    best_dx, best_dy = np.unravel_index(np.argmax(C, axis=None), C.shape)
    return (best_dx, best_dy)
    
def transform_dx(x, n):
    return (x - n // 2) % n - n // 2 - n % 2


def transform_coord(origin_x, x, n):
    return -transform_dx(x, n) + origin_x


# image = imread("tests/00_test_img_input/img.png")
# source_img = np.array(image)

def align(source_img, green_coord):
    green_x, green_y = green_coord
    canals, h, w = crop(source_img) #shape (n, m, 3)
    canals_copy = copy(canals)
    n, m = canals.shape[:2]
    blue_dx, blue_dy = find_shift(canals[..., 1], canals[... , 0])
    red_dx, red_dy = find_shift(canals[..., 1], canals[... , 2])
    blue_x = transform_coord(green_x, blue_dx, n) - h
    blue_y = transform_coord(green_y, blue_dy, m)
    red_x = transform_coord(green_x, red_dx, n) + h
    red_y = transform_coord(green_y, red_dy, m)
    final_img = np.copy(canals)
    final_img[... , 2] = np.roll(canals[... , 0], (transform_dx(blue_dx, n), transform_dx(blue_dy, m)), axis=(0, 1))
    final_img[... , 0] = np.roll(canals[... , 2], (transform_dx(red_dx, n), transform_dx(red_dy, m)), axis =(0, 1))
    return final_img, (blue_x, blue_y), (red_x, red_y)
    