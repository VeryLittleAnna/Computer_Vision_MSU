import numpy as np
from math import ceil, log10


filters = [
    np.array([[0, 0, -1, 0, 0], 
    [0, 0, 2, 0, 0],
    [-1, 2, 4, 2, -1],
    [0, 0, 2, 0, 0],
    [0, 0, -1, 0, 0]], dtype=np.float64),
    
    np.array([[0, 0, 0.5, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 0.5, 0, 0]], dtype=np.float64),

    np.array([[0, 0, -1, 0, 0],
    [0, -1, 4, -1, 0],
    [0.5, 0, 5, 0, 0.5],
    [0, -1, 4, -1, 0],
    [0, 0, -1, 0, 0]], dtype=np.float64),

    np.array([[0, 0, -1.5, 0, 0],
    [0, 2, 0, 2, 0], 
    [-1.5, 0, 6, 0, -1.5],
    [0, 2, 0, 2, 0],
    [0, 0, -1.5, 0, 0]], dtype=np.float64)
]


def get_bayer_masks(n_rows, n_cols):
    red = np.tile(np.array([[0, 1], [0, 0]], dtype=bool), (ceil(n_rows / 2), ceil(n_cols / 2)))[:n_rows, :n_cols,]
    green = np.tile(np.array([[1, 0], [0, 1]], dtype=bool), (ceil(n_rows / 2), ceil(n_cols / 2)))[:n_rows, :n_cols,]
    blue = np.tile(np.array([[0, 0], [1, 0]], dtype=bool), (ceil(n_rows / 2), ceil(n_cols / 2)))[:n_rows, :n_cols,]
    return np.dstack([red, green, blue])


def get_colored_img(raw_img):
    """
    raw_img - shape (n, m)
    """
    mask = get_bayer_masks(*raw_img.shape)
    return np.dstack([raw_img * mask[...,i] for i in range(3)])

def bilinear_interpolation(colored_img):
    """
    colored_img - shape (n, m, 3)
    """
    n, m = colored_img.shape[:2]
    mask = get_bayer_masks(n, m)
    final_img = np.array(colored_img)
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            for c in range(3):
                if (not((c == 1 and (i + j) % 2 == 0) or (c == 0 and i % 2 == 0 and  j % 2 != 0) or (c == 2 and i % 2 != 0 and j % 2 == 0))):
                    cnt = (4 if ((i + j) % 2 == 1 or c == 1) else 2)
                    cur_sum = 0
                    for ii in range(i - 1, i + 2):
                        for jj in range(j - 1, j + 2):
                            cur_sum += colored_img[ii, jj, c]
                    final_img[i, j, c] = cur_sum // cnt
                    # final_img[i, j, c] = np.sum(colored_img[i-1:i+2, j-1:j+2, c]) //cnt # [mask[i-1:i+2, j-1:j+2, c]])
    return final_img
   

def improved_interpolation(raw_img):
    """
    raw_img - shape (n, m)
    """
    n, m = raw_img.shape
    raw_img = raw_img.astype(np.float64)
    global filters
    for i in filters:
        i /= np.sum(i)
    colored_img = get_colored_img(raw_img)
    final_img = np.array(colored_img)
    indices = [
        [[1, None, 2], [None, 0, 3]],
        [[3, 0, None], [2, None, 1]]
    ]
    for i in range(2, n - 2):
        for j in range(2, m - 2):
            for c in range(3):
                ind = indices[i % 2][j % 2][c]
                if (ind != None):
                    # print(raw_img[i-2:i+3, j-2:j+3].shape, "vs", filters[ind].shape)
                    final_img[i, j, c] = np.sum(raw_img[i-2:i+3, j-2:j+3] * filters[ind])
    return np.array(final_img.clip(0, 255), dtype='uint8')


def compute_psnr(img_pred, img_gt):
    """
    img_pred, img_gt - shape (n, m, c)
    """
    n, m, c = img_pred.shape
    img_pred = np.array(img_pred, dtype=np.float64)
    img_gt = np.array(img_gt, dtype=np.float64)
    mse = np.sum(np.square(img_pred - img_gt)) / n / m / c
    if (mse == 0):
        raise ValueError 
    psnr = 10 * log10(np.max(img_gt) ** 2 / mse)
    return psnr