from xml.dom.expatbuilder import theDOMImplementation
import numpy as np
from scipy.fft import fft2, ifft2


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    ans = np.zeros((size, size), dtype=np.float64)
    n = size // 2
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            r = np.sqrt(i ** 2 + j ** 2)
            ans[i + n, j + n] = 1 / (2 * np.pi * sigma ** 2) * np.exp(- r * r / (2 * sigma ** 2))
    return ans / np.sum(ans)


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    H = np.pad(h, ((0, shape[0] - h.shape[0]), (0, shape[1] - h.shape[1])), 'constant', constant_values=(0, 0))
    return fft2(H)


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    mask = (np.absolute(H, dtype=np.float64) <= threshold).astype(np.complex64)
    d = (1 - mask) * H + mask * np.ones_like(H)
    H_inv = (1 - mask) / d 
    return H_inv.astype(np.complex64)


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    F_est = fourier_transform(blurred_img, blurred_img.shape) * inverse_kernel(fourier_transform(h, blurred_img.shape), threshold)
    f_est = ifft2(F_est)
    ans = np.absolute(f_est, dtype=np.float64)
    return ans


def wiener_filtering(blurred_img, h, K=0.00067):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    H_conj = np.conj(H)
    HH = H_conj * H
    F_est = H_conj / (HH + K) * G
    f_est = np.absolute(ifft2(F_est), dtype=np.float64)
    return f_est


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    mse = np.square(img1 - img2).mean()
    return 20 * np.log10(255 / np.sqrt(mse))