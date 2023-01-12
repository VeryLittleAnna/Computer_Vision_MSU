import numpy as np
from copy import copy

def calc_grad(Y):
    gradX = np.zeros(Y.shape, dtype=np.float64)
    gradY = np.zeros(Y.shape, dtype=np.float64)
    n, m = Y.shape
    for y in range(m):
        gradX[0, y] = Y[1, y] - Y[0, y]
        gradX[n - 1, y] = Y[n - 1, y] - Y[n - 2, y]
        for x in range(1, n - 1):
            gradX[x, y] = Y[x + 1, y] - Y[x - 1, y]
    
    for x in range(n):
        gradY[x, 0] = Y[x, 1] - Y[x, 0]
        gradY[x, m - 1] = Y[x, m - 1] - Y[x, m - 2]
        for y in range(1, m - 1):
            gradY[x, y] = Y[x, y + 1] - Y[x, y - 1]

    grad = np.sqrt(np.square(gradX) + np.square(gradY))
    return grad

def find_seam(img, mask, Y, grad):
    n, m = Y.shape
    energy = np.zeros(Y.shape, dtype=np.float64)
    mask_seam = np.zeros(Y.shape, dtype=np.uint8)
    energy[0, :] = grad[0, :]
    energy += mask.astype(np.float64) * 256 * n * m
    for x in range(1, n):
        energy[x, 0] += min(energy[x - 1, 0], energy[x - 1, 1]) + grad[x, 0]
        energy[x, m - 1] += min(energy[x - 1, m - 1], energy[x - 1, m - 2]) + grad[x, m - 1]
        for y in range(1, m - 1):
            energy[x, y] += min(energy[x - 1, y - 1], energy[x - 1, y], energy[x - 1, y + 1]) + grad[x, y]

    cur_j = np.argmin(energy[n - 1, :])
    cur_i = n - 1
    mask_seam[n - 1, cur_j] = 1
    while cur_i > 0:
        cur_i -= 1
        #cur_j += - 1 + np.argmin(energy[cur_i, max(0, cur_j - 1):min(m, cur_j + 2)])
        cur_j += (0 if cur_j == 0 else -1) + np.argmin(energy[cur_i, max(0, cur_j - 1):min(m, cur_j + 2)])
        mask_seam[cur_i, cur_j] = 1
    return mask_seam


def horiz_shrink(img, mask, Y, grad):
    mask_seam = find_seam(img, mask, Y, grad)
    ans = np.zeros((Y.shape[0], Y.shape[1] - 1, 3), dtype=np.uint8)
    ans[..., 0] = np.array([np.delete(i, np.argmax(mask_seam[pos, :]), axis=0) for pos, i in enumerate(img[..., 0])], dtype=np.uint8)
    ans[..., 1] = np.array([np.delete(i, np.argmax(mask_seam[pos, :]), axis=0) for pos, i in enumerate(img[..., 1])], dtype=np.uint8)
    ans[..., 2] = np.array([np.delete(i, np.argmax(mask_seam[pos, :]), axis=0) for pos, i in enumerate(img[..., 2])], dtype=np.uint8)
    ans_mask = np.array([np.delete(i, np.argmax(mask_seam[pos, :]), axis=0) for pos, i in enumerate(mask)], dtype=np.int8)
    return ans, ans_mask, mask_seam

def horiz_expand(img, mask, Y, grad):
    mask_seam = find_seam(img, mask, Y, grad)
    ans = np.zeros((Y.shape[0], Y.shape[1] + 1, 3), dtype=np.uint8)
    ans[..., 0] = np.array([np.insert(i, (ind := np.argmax(mask_seam[pos, :])), (i[ind] + i[min(ind + 1, Y.shape[1] - 1)]) // 2, axis=0) for pos, i in enumerate(img[..., 0])], dtype=np.uint8)
    ans[..., 1] = np.array([np.insert(i, (ind := np.argmax(mask_seam[pos, :])), (i[ind] + i[min(ind + 1, Y.shape[1] - 1)]) // 2, axis=0) for pos, i in enumerate(img[..., 1])], dtype=np.uint8)
    ans[..., 2] = np.array([np.insert(i, (ind := np.argmax(mask_seam[pos, :])), (i[ind] + i[min(ind + 1, Y.shape[1] - 1)]) // 2, axis=0) for pos, i in enumerate(img[..., 2])], dtype=np.uint8)
    ans_mask = np.array([np.insert(i, np.argmax(mask_seam[pos, :]), 1, axis=0) for pos, i in enumerate(mask)], dtype=np.int8)
    return ans, ans_mask, mask_seam

def seam_carve(img, mode, mask=None):
    if (mask is None):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int8) 
    img = img.astype(dtype=np.float64)
    copy_img = copy(img)
    Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    grad = calc_grad(Y)
    if (mode == "horizontal shrink"):
        return horiz_shrink(copy_img, mask, Y, grad)
    elif (mode == "horizontal expand"):
        return horiz_expand(copy_img, mask, Y, grad)
    else:
        img_t = np.dstack([np.transpose(copy_img[:,:,0]), np.transpose(copy_img[:,:,1]), np.transpose(copy_img[:,:,2])])
        mask_t = np.transpose(mask)
        Y_t = np.transpose(Y)
        grad_t = np.transpose(grad) 
        if (mode == "vertical shrink"):
            ans_t, mask_t, mask_seam_t = horiz_shrink(img_t, mask_t, Y_t, grad_t)
        else:
            ans_t, mask_t, mask_seam_t = horiz_expand(img_t, mask_t, Y_t, grad_t)
        ans = np.dstack([np.transpose(ans_t[:,:,0]), np.transpose(ans_t[:,:,1]), np.transpose(ans_t[:,:,2])])
        return ans, np.transpose(mask_t), np.transpose(mask_seam_t)
    
    
                       
       