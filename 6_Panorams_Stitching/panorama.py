from turtle import right
import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv, svd

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=500):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    # img = rgb2gray(img)

    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    img = rgb2gray(img)
    descriptor_extractor.detect_and_extract(img)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    return keypoints, descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """
    N = points.shape[0]
    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    matrix = np.zeros((3, 3))
    Cx = np.mean(pointsh[0, ...])
    Cy = np.mean(pointsh[1, ...])
    Norm = N * np.sqrt(2) / np.sqrt(np.sum((np.square(pointsh[0, ...] - Cx) + np.square(pointsh[1, ...] - Cy))))
    matrix = np.array([
        [Norm, 0, - Norm * Cx],
        [0, Norm, - Norm * Cy],
        [0, 0, 1]
    ], dtype=np.float64)
    res = matrix @ pointsh
    return matrix, (matrix @ pointsh)[:2, ...].T


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    H = np.zeros((3, 3))
    n = src_keypoints.shape[0]
    # N = np.square(np.array(np.ceil(np.sqrt(2 * n)), dtype=np.int8))
    N = 2 * n
    A = np.zeros((N, 9), dtype=np.float64)
    src_pointsh = np.row_stack([src.T, np.ones((src.shape[0]))])
    A[:2*n:2,...] = np.concatenate(((-src_pointsh).T, np.zeros((n, 3)), src_pointsh.T * dest[..., 0].reshape(1, n).T), axis=1)
    A[1:2*n:2, ...] = np.concatenate((np.zeros((n, 3)), (-src_pointsh).T, src_pointsh.T * dest[..., 1].reshape(1, n).T), axis=1)
    u, s, vH = svd(A, full_matrices=True)
    h = vH[-1, ...]
    H = h.reshape(3,3)
    return inv(dest_matrix) @ H @ src_matrix


def distance(a, b):
    """
    a ((N, 2), np.ndarray):
    b ((N, 2), np.ndarray)
    """
    a.astype(np.float64)
    b.astype(np.float64)
    return np.sqrt(np.sum(np.square(a - b), axis=1))


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=1000, residual_threshold=2, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """
    matches = match_descriptors(src_descriptors, dest_descriptors)
    N = matches.shape[0]
    # matches = np.stack([np.arange(N), np.arange(N)], axis=0)
    src_keypoints, dest_keypoints = src_keypoints[matches[..., 0]], dest_keypoints[matches[..., 1]]
    inliers_mask, leader_cnt_inliers = None, -1
    for tt in range(max_trials):
        points = np.random.choice(np.arange(N), size=4, replace=False)
        H = find_homography(src_keypoints[points], dest_keypoints[points])
        transform = ProjectiveTransform(H)
        src_points_transform = transform(src_keypoints)
        cur_inliers_mask = distance(src_points_transform, dest_keypoints) < residual_threshold
        cur_cnt_inliers = np.sum(cur_inliers_mask)
        if cur_cnt_inliers >= leader_cnt_inliers:
            leader_cnt_inliers = cur_cnt_inliers
            inliers_mask = cur_inliers_mask
    points = inliers_mask
    if return_matches:
        return ProjectiveTransform(find_homography(src_keypoints[points], dest_keypoints[points])), matches[inliers_mask]
    return ProjectiveTransform(find_homography(src_keypoints[points], dest_keypoints[points]))


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [DEFAULT_TRANSFORM] * image_count
    result[center_index] = DEFAULT_TRANSFORM()

    for i in range(center_index - 1, -1, -1):
        result[i] = result[i + 1] + forward_transforms[i]
    for i in range(center_index + 1, image_count):
        result[i] = result[i - 1] + forward_transforms[i - 1].inverse
    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
            не только...
        """
    #simple_center_warps = find_simple_center_warps(simple_center_warps)
    corners = tuple(get_corners(image_collection, simple_center_warps))
    min_coords, max_coords = get_min_max_coords(corners)
    shift = np.array([
        [1, 0, max(0, -min_coords[1])],
        [0, 1, max(0, -min_coords[0])],
        [0, 0, 1]
    ])
    shift_transform = ProjectiveTransform(shift)
    final_transforms = []
    for i in range(len(corners)):
        final_transforms += [simple_center_warps[i] + shift_transform]
    return tuple(final_transforms), (int(max_coords[1] - min_coords[1]), int(max_coords[0] - min_coords[0]))


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    transform = rotate_transform_matrix(transform).inverse
    result_image = warp(image, transform, output_shape=output_shape)
    mask = np.ones(image.shape[:2], dtype=np.bool8)
    mask = warp(mask, transform, output_shape=output_shape)
    return result_image, mask.astype(np.bool8)


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)
    for i in range(len(image_collection)):
        img, mask = warp_image(image_collection[i], final_center_warps[i], output_shape)
        result_mask += mask
        result[mask.astype(np.bool8)] = img[mask.astype(np.bool8)]
    return result
    


def get_gaussian_pyramid(image, n_layers=4, sigma=1):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    result = [image]
    for i in range(n_layers - 1):
        result.append(gaussian(result[-1], sigma))
    return tuple(result)


def get_laplacian_pyramid(image, n_layers=4, sigma=1):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    gaussian_pyramid = get_gaussian_pyramid(image, n_layers=n_layers, sigma=sigma)
    result = [gaussian_pyramid[i] - gaussian_pyramid[i + 1] for i in range(n_layers - 1)] + [gaussian_pyramid[-1]]
    return tuple(result)


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=4, image_sigma=1, merge_sigma=6):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    N = len(image_collection)
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)
    warps = [list(warp_image(image_collection[i], final_center_warps[i], output_shape)) for i in range(len(image_collection))]
    for i in range(N - 1):
        mask_inter = (warps[i][1] & warps[i + 1][1]).astype(np.bool8)
        along_x = np.argwhere(np.any(mask_inter, axis=0))
        if along_x.shape[0] == 0:
            min_y, max_y = 0, warps[i][1].shape[1]
        else:
            min_y, max_y = np.min(along_x, axis=None), np.max(along_x)
        mid_y = (min_y + max_y) // 2
        warps[i][1][..., mid_y:] = 0
        warps[i + 1][1][..., :mid_y] = 0
    for i in range(N):
        warps[i][1].astype(np.float32)
        masks = get_gaussian_pyramid(warps[i][1], n_layers, merge_sigma)
        imgs = get_laplacian_pyramid(warps[i][0], n_layers, image_sigma)
        total_img = np.zeros(warps[i][0].shape)
        for i in range(n_layers):
            for c in range(3):
                total_img[..., c] = masks[i] * imgs[i][..., c]
        total_mask = sum(masks)
        # result_mask += total_mask
        result += total_img
    result = np.clip(result, 0, 1)
    return result       

def cylindrical_inverse_map(coords, h, w, scale):
    """Function that transform coordinates in the output image
    to their corresponding coordinates in the input image
    according to cylindrical transform.

    Use it in skimage.transform.warp as `inverse_map` argument

    coords ((M, 2) np.ndarray) : coordinates of output image (M == col * row)
    h (int) : height (number of rows) of input image
    w (int) : width (number of cols) of input image
    scale (int or float) : scaling parameter

    Returns:
        (M, 2) np.ndarray : corresponding coordinates of input image (M == col * row) according to cylindrical transform
    """
    # your code here
    pass

def warp_cylindrical(img, scale=None, crop=True):
    """Warp image to cylindrical coordinates

    img ((H, W, 3)  np.ndarray) : image for transformation
    scale (int or None) : scaling parameter. If None, defaults to W * 0.5
    crop (bool) : crop image to fit (remove unnecessary zero-padding of image)

    Returns:
        (H, W, 3)  np.ndarray : warped image (H and W may differ from original)
    """
    # your code here
    pass


# Pick a good scale value for the 5 test image sets
cylindrical_scales = {
    0: None,
    1: None,
    2: None,
    3: None,
    4: None,
}
