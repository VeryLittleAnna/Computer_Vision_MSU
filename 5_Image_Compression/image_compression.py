from bz2 import decompress
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы и проекция матрицы на новое пр-во
    """
    # Отцентруем каждую строчку матрицы
    matrix = matrix.astype(np.float64)
    m = np.mean(matrix, axis = 1)[:, None]
    matrix[:] -= m
    # Найдем матрицу ковариации
    cov = np.cov(matrix)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_val, eig_vec = np.linalg.eigh(cov)
    # Посчитаем количество найденных собственных векторов
    eig_vec_cnt = eig_vec.shape[1]
    # Сортируем собственные значения в порядке убывания
    s = np.argsort(eig_val)[::-1]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    eig_vec = eig_vec[:, s]
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    # Оставляем только p собственных векторов
    eig_vec = eig_vec[:, :p]
    # Проекция данных на новое пространство
    return eig_vec, eig_vec.T @ matrix, m[:, 0]


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        eig_vec, projections, means = comp
        result_img.append(eig_vec @ projections + means[:, None])
    return np.clip(np.dstack(result_img), 0, 255).astype(np.uint8)


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[... , j], p))
        decompressed = pca_decompression(compressed)
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")

RGBtoYCbCr = np.array([[0.299, 0.587, 0.114],
                        [-0.1687, -0.3313, 0.5],
                        [0.5, -0.4187, -0.0813]], dtype=np.float64)

# RGBshift = np.array([[0], [128], [128]], dtype=np.float64)

YCbCrtoRGB = np.array([[1, 0, 1.402],
                        [1, -0.34414, -0.71414],
                        [1, 1.77, 0]], dtype=np.float64)


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    print(img[..., np.newaxis].shape)
    print(RGBtoYCbCr.shape)
    ans =  np.matmul(RGBtoYCbCr, img[..., np.newaxis])[..., 0] #(3, 3)  (n, m, 3, 1) -> (n, m, 3, 1)
    ans[..., 1:3] += 128
    return np.array(np.clip(ans, 0, 255), dtype=np.uint8)


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    ans = np.array(img, dtype=np.float64)
    ans[..., 1:3] -= 128
    ans = np.matmul(YCbCrtoRGB, ans[..., np.newaxis])[..., 0]
    return np.array(np.clip(ans, 0, 255), dtype=np.uint8)


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[:, :, 1] = gaussian_filter(ycbcr_img[:, :, 1], 10)
    ycbcr_img[:, :, 2] = gaussian_filter(ycbcr_img[:, :, 2], 10)
    res_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(res_img)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[:, :, 0] = gaussian_filter(ycbcr_img[:, :, 0], 10)
    res_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(res_img)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    a = gaussian_filter(component, 10)
    return a[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    G = np.zeros((8, 8), dtype=np.float64)
    block = block.astype(dtype=np.float64)
    for u in range(8):
        for v in range(8):
            for x in range(8):
                for y in range(8):
                    G[u, v] += block[x, y] * np.cos((2.0 * x + 1) * u * np.pi / 16) * np.cos((2.0 * y + 1) * v * np.pi / 16) 
            G[u, v] *= 0.25 * (1 / np.sqrt(2) if u == 0 else 1) * (1 / np.sqrt(2) if v == 0 else 1)
    return G


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

quantization_matrices = [y_quantization_matrix, color_quantization_matrix]


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    return np.round(block / quantization_matrix)



def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100
    if q < 50:
        s = 5000 // q
    elif q <= 99:
        s = 200 - 2 * q
    else:
        s = 1
    res = np.floor((50 + s * default_quantization_matrix) / 100)
    res += np.ones((8, 8)) * (res == 0)
    return res

mask_zigzag = np.array([[ 0,  1,  5,  6, 14, 15, 27, 28],
       [ 2,  4,  7, 13, 16, 26, 29, 42],
       [ 3,  8, 12, 17, 25, 30, 41, 43],
       [ 9, 11, 18, 24, 31, 40, 44, 53],
       [10, 19, 23, 32, 39, 45, 52, 54],
       [20, 22, 33, 38, 46, 51, 55, 60],
       [21, 34, 37, 47, 50, 56, 59, 61],
       [35, 36, 48, 49, 57, 58, 62, 63]], dtype=np.uint8)

def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    res = [0] * 64
    for i in range(8):
        for j in range(8):
            res[mask_zigzag[i, j]] = int(block[i, j])
    return res


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    ans = list()
    prev, cnt = 0, 0
    for i in range(len(zigzag_list)):
        if (zigzag_list[i] == 0):
            cnt += 1
        else:
            if (cnt != 0):
                ans.append(0)
                ans.append(cnt)
                cnt = 0
            ans.append(zigzag_list[i])
    if (cnt != 0):
        ans.append(0)
        ans.append(cnt)
    return ans


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    # Переходим из RGB в YCbCr
    ycbcr_img = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    comp = [ycbcr_img[..., i] for i in range(3)]
    comp[1] = downsampling(comp[1])
    comp[2] = downsampling(comp[2])
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    result_list = [[], [], []]
    for c in range(3):
        comp[c] = (comp[c].astype(np.int16) - 128).astype(np.int8)
        for i in range(0, comp[c].shape[0], 8):
            for j in range(0, comp[c].shape[1], 8):
                result_list[c].append(compression(zigzag(quantization(dct(comp[c][i:i+8, j:j+8]), quantization_matrixes[0 if c == 0 else 1]))))
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    return result_list


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    i = 0
    ans = []
    while i < len(compressed_list):
        if (compressed_list[i] == 0):
            ans += [0] * compressed_list[i + 1]
            i += 1
        else:
            ans.append(compressed_list[i])
        i += 1
    return ans


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    global mask_zigzag
    ans = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            ans[i, j] = input[mask_zigzag[i, j]]
    return ans


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    res = block * quantization_matrix
    return res


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    G = np.zeros((8, 8), dtype=np.float64)
    block = block.astype(dtype=np.float64)
    for x in range(8):
        for y in range(8):
            for u in range(8):
                for v in range(8):
                    G[x, y] += (1 / np.sqrt(2) if u == 0 else 1) * (1 / np.sqrt(2) if v == 0 else 1) * block[u, v] * np.cos((2.0 * x + 1) \
                             * u * np.pi / 16) * np.cos((2.0 * y + 1) * v * np.pi / 16) 
            G[x, y] /= 4
    return np.round(G)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    a, b = component.shape[:2]
    res = np.zeros((2 * a, 2 * b), dtype=np.float64)
    res[::2, ::2] = component
    res[1::2, ::2] = component
    res[::2, 1::2] = component
    res[1::2, 1::2] = component
    return res



def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    a = [[], [], []]
    n = result_shape[0]
    m = result_shape[0]
    a = [np.zeros((n, m), dtype=np.float64)] + [np.zeros((n // 2, m // 2), dtype=np.float64) for i in range(2)]    
    for i in range(n // 16):
        for j in range(m // 16):
            for c in range(1, 3):
                ind = 1
                if (i * (m // 16) + j >= len(result[c])):
                    print(i, j, len(result[c]), n, m)
                a[c][i*8:(i+1)*8, j*8:(j+1)*8] = inverse_dct(inverse_quantization(inverse_zigzag
                        (inverse_compression(result[c][i * (m // 16) + j])),  quantization_matrixes[ind]) )
    for i in range(n // 8):
        for j in range(m // 8):
            c = 0
            ind = 0
            if (i * (m // 8) + j >= len(result[c])):
                print(i, j, len(result[c]), n // 8, m // 8)
            a[c][i*8:(i+1)*8, j*8:(j+1)*8] = inverse_dct(inverse_quantization(inverse_zigzag
                    (inverse_compression(result[c][i * (m // 8) + j])),  quantization_matrixes[ind]) )
    a[0] += 128
    a[1] += 128
    a[2] += 128
    a[1] = upsampling(a[1])
    a[2] = upsampling(a[2])
    ycbcr_img = np.dstack(a)
    img = ycbcr2rgb(ycbcr_img)
    return img

def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    print("AAAAA: ", img.shape)
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        matr = [own_quantization_matrix(y_quantization_matrix, p), own_quantization_matrix(color_quantization_matrix, p)]
        compressed = jpeg_compression(img, matr)
        decomp = jpeg_decompression(compressed, img.shape, matr)

        axes[i // 3, i % 3].imshow(decomp)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")



def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")

