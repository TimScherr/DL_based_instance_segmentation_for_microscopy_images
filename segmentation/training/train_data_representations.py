import cv2
from itertools import product

import numpy as np
import os
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt, grey_closing, generate_binary_structure, distance_transform_cdt
from skimage import measure, morphology
from skimage.morphology import disk, remove_small_objects
from segmentation.utils.utils import get_nucleus_ids


def get_label(mask, label_type, max_mal):
    """ Calculate training data representation / label

    :param mask:
    :param label_type:
    :param max_mal:
    :return:
    """

    if label_type == 'boundary':
        label = boundary_label(mask)
    elif label_type == 'border':
        label = border_label(mask)
    elif label_type == 'adapted_border':
        label = adapted_border_label(mask)
    elif label_type == 'j4':
        label = j4_label(mask)
    elif label_type == 'jcell':
        label = jcell_label(mask)
    elif label_type == 'dist':
        label = dist(mask, search_radius=int(np.ceil(0.75 * max_mal)))
    elif label_type == 'cell_dist':
        label = cell_distance_label(mask, search_radius=int(np.ceil(0.75 * max_mal)))
    elif label_type == 'cell_dist_clipped':
        label = cell_distance_label(mask, search_radius=int(np.ceil(0.75 * max_mal)), apply_clipping=True)
    elif label_type == 'distance':
        label = distance_label(mask, search_radius=int(np.ceil(0.75 * max_mal)))
    else:
        raise Exception('Label type not known')

    return label


def bottom_hat_closing(label):
    """ Bottom-hat-transform based grayscale closing.

    :param label: Intensity coded label image.
        :type label:
    :return: closed label (only closed regions, all 1), closed label (only closed regions, 0.8-1.0)
    """

    label_bin = np.zeros_like(label, dtype=bool)

    # Apply closing to each nucleus to avoid artifacts
    nucleus_ids = get_nucleus_ids(label)
    for nucleus_id in nucleus_ids:
        nucleus = (label == nucleus_id)
        nucleus = ndimage.binary_closing(nucleus, disk(3))
        label_bin[nucleus] = True

    # Bottom-hat-transform
    label_bottom_hat = ndimage.binary_closing(label_bin, disk(3)) ^ label_bin
    label_closed = (~label_bin) & label_bottom_hat

    # Integrate gaps better into the neighbor distances
    label_closed = measure.label(label_closed.astype(np.uint8))
    props = measure.regionprops(label_closed)
    label_closed_corr = (label_closed > 0).astype(np.float32)
    for i in range(len(props)):
        if props[i].minor_axis_length >= 3:
            single_gap = label_closed == props[i].label
            single_gap_border = single_gap ^ ndimage.binary_erosion(single_gap, generate_binary_structure(2, 1))
            label_closed_corr[single_gap] = 1
            label_closed_corr[single_gap_border] = 0.8  # gets scaled later to 0.84

    return label_closed, label_closed_corr


def boundary_label(label):
    """ Boundary label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :return: Boundary label image.
    """

    label_bin = label > 0

    kernel = np.ones(shape=(3, 3), dtype=np.uint8)

    # Pre-allocation
    boundary = np.zeros(shape=label.shape, dtype=bool)

    nucleus_ids = get_nucleus_ids(label)

    for nucleus_id in nucleus_ids:
        nucleus = (label == nucleus_id)
        nucleus_boundary = ndimage.binary_dilation(nucleus, kernel) ^ nucleus
        boundary += nucleus_boundary

    label_boundary = np.maximum(label_bin, 2 * boundary)

    return label_boundary.astype(np.uint8)


def border_label(label):
    """ Border label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :return: Border label image.
    """

    label_bin = label > 0
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)

    # Pre-allocation
    boundary = np.zeros(shape=label.shape, dtype=bool)

    nucleus_ids = get_nucleus_ids(label)

    for nucleus_id in nucleus_ids:
        nucleus = (label == nucleus_id)
        nucleus_boundary = ndimage.binary_dilation(nucleus, kernel) ^ nucleus
        boundary += nucleus_boundary

    border = boundary ^ (ndimage.binary_dilation(label_bin, kernel) ^ label_bin)
    label_border = np.maximum(label_bin, 2 * border)

    return label_border.astype(np.uint8)


def adapted_border_label(label):
    """ Adapted border label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :return: Adapted border label image.
    """

    # Instances not necessarily counted starting from 1
    label = measure.label(label, background=0)

    if len(get_nucleus_ids(label)) > 255:
        # Avoid conversion of 255 to 0
        label[label > 255] += 1
        if len(get_nucleus_ids(label)) > 510:
            label[label > 510] += 1
        # raise Exception('Canny method works only with uint8 images but more than 255 nuclei detected.')

    kernel = np.ones(shape=(3, 3), dtype=np.uint8)

    label_bin = label > 0

    boundary = cv2.Canny(label.astype(np.uint8), 1, 1) > 0

    border = cv2.Canny(label_bin.astype(np.uint8), 1, 1) > 0
    border = boundary ^ border

    border_adapted = ndimage.binary_dilation(border.astype(np.uint8), kernel)
    cell_adapted = ndimage.binary_erosion(label_bin.astype(np.uint8), kernel)

    border_adapted = ndimage.binary_closing(border_adapted, kernel)
    label_adapted_border = np.maximum(cell_adapted, 2 * border_adapted)

    return label_adapted_border.astype(np.uint8)


def dist(label, search_radius):
    """ Naylor DIST method
    """

    # Preallocation
    label_dist = np.zeros(shape=label.shape, dtype=np.float)

    # Find centroids, crop image, calculate distance transforms
    props = measure.regionprops(label)
    for i in range(len(props)):

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - search_radius, 0)):int(min(centroid[0] + search_radius, label.shape[0])),
                       int(max(centroid[1] - search_radius, 0)):int(min(centroid[1] + search_radius, label.shape[1]))
                       ]
        nucleus_crop_dist = distance_transform_cdt(nucleus_crop)

        label_dist[
        int(max(centroid[0] - search_radius, 0)):int(min(centroid[0] + search_radius, label.shape[0])),
        int(max(centroid[1] - search_radius, 0)):int(min(centroid[1] + search_radius, label.shape[1]))
        ] += nucleus_crop_dist

    return label_dist.astype(np.float32)


def j4_label(label, k_neighbors=2, se_radius=4):
    """ Pena label creation for the J4 method (background, cell, touching, gap).

    Reference: Pena et al. "J regularization improves imbalanced mutliclass segmentation". In: 2020 IEEE 17th
        International Symposium on Biomedical Imaging (ISBI). 2020.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param k_neighbors: Neighborhood parameter needed for the creation of the touching class.
        :type k_neighbors: int
    :param se_radius: Structuring element (hypersphere) radius needed for the creation of the gap class.
    :return: Pena/J4 label image.
    """

    # Bottom hat transformation:
    label_bin = label > 0
    se = disk(se_radius)
    label_bottom_hat = ndimage.binary_closing(label_bin, se) ^ label_bin

    neighbor_mask = compute_neighbor_instances(label, k_neighbors)

    label_bg = (~label_bin) & (~label_bottom_hat)
    label_gap = (~label_bin) & label_bottom_hat
    label_touching = label_bin & (neighbor_mask > 1)
    label_cell = ~(label_bg | label_gap | label_touching)

    # remove some artifacts in gap class
    gap_touching = label_gap | label_touching
    gap_touching_corr = remove_small_objects(morphology.label(gap_touching), min_size=8, connectivity=4) > 0
    label_bg[label_gap ^ gap_touching_corr] = True
    label_gap = label_gap & gap_touching_corr

    # 0: background, 1: cell, 2: touching, 3: gap
    label_j4 = np.maximum(label_bg, 2 * label_cell)
    label_j4 = np.maximum(label_j4, 3 * label_touching)
    label_j4 = np.maximum(label_j4, 4 * label_gap)
    label_j4 -= 1

    return label_j4.astype(np.uint8)


def compute_neighbor_instances(instance_mask, k_neighbors):
    """ Function to find instances in the neighborhood. """
    indices = [list(range(s)) for s in instance_mask.shape]

    mask_shape = instance_mask.shape
    padded_mask = np.pad(instance_mask, pad_width=k_neighbors, constant_values=0)
    n_neighbors = np.zeros_like(instance_mask)

    crop_2d = lambda x, y: (slice(x[0], y[0]), slice(x[1], y[1]))
    crop_3d = lambda x, y: (slice(x[0], y[0]), slice(x[1], y[1]), slice(x[2], y[2]))
    if len(mask_shape) == 2:
        crop_func = crop_2d
    elif len(mask_shape) == 3:
        crop_func = crop_3d
    else:
        raise AssertionError(f'instance mask shape is not 2 or 3 dimensional {instance_mask.shape}')

    for index in product(*indices):
        top_left = np.array(index) - k_neighbors + k_neighbors  # due to shift from padding
        bottom_right = np.array(index) + 2 * k_neighbors + 1
        crop_box = crop_func(top_left, bottom_right)
        crop = padded_mask[crop_box]
        n_neighbors[index] = len(set(crop[crop > 0]))

    return n_neighbors


def jcell_label(mask):
    """ j4 label from https://pypi.org/project/jcell-ISBI/#description """

    return inst2sem(mask)


def inst2sem(L, se1_radius=2, se2_radius=4):
    """
    2D Instance to 4 classes semantic label transformation.
    The label file is expected to be an 8bit/16bits 1-channel image with where each
    object has their own identifier.
    The output will be an 8bits 1-channel image with semantic classes.

    https://pypi.org/project/jcell-ISBI/#description
    """
    if L.ndim == 3:
        L = L[:, :, 0]

    H = inst2sem2d(L, se1_radius)
    if se2_radius > 0:
        H = sem32sem42d(H, se2_radius)

    return H.astype("uint8")


def inst2sem2d(inst, se=3):
    """
    Instance to 3 classes semantic label transformation

    https://pypi.org/project/jcell-ISBI/#description
    """
    inst = np.lib.pad(
        inst, ((se, se), (se, se)), "constant", constant_values=0
    )
    gt = (inst > 0).astype("uint8")

    # @jit_filter_function
    def filter(neigh):
        center = len(neigh) // 2
        return (
            (neigh[center] != 0) * (neigh[center] != neigh) * (neigh != 0)
        ).sum()

    se2 = se if se >= 3 else 3
    I2 = ndimage.generic_filter(inst, filter, size=se2)
    I2[I2 > 0] = 1
    if se > 0:
        I2 = ndimage.binary_dilation(I2 > 0, structure=disk(se))
    gt[I2 > 0] = 2

    if se > 0:
        gt = gt[se:-se, se:-se]
    return gt


def sem32sem42d(S, se):
    """
    3 semantic classes to 4 semantic classes transformation

    https://pypi.org/project/jcell-ISBI/#description
    """
    S = np.lib.pad(S, ((se, se), (se, se)), "constant", constant_values=0)

    B = 1 - (S == 0).astype("uint8")
    B = ndimage.binary_closing(B, structure=disk(se)).astype("uint8") - B
    B = remove_small_objects(morphology.label(B), min_size=8, connectivity=4)
    S[B >= 1] = 3

    if se > 0:
        S = S[se:-se, se:-se]
    return S


def cell_distance_label(label, search_radius, apply_clipping=False, clip_val=5):
    """ Cell and neigbhor distance label creation (Euclidean distance).

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param search_radius: Defines the area to compute the distance transform for the cell distances.
        :type search_radius: int
    :return: Cell distance label image, neighbor distance label image.
    """

    # Preallocation
    label_dist = np.zeros(shape=label.shape, dtype=np.float)

    # Find centroids, crop image, calculate distance transforms
    props = measure.regionprops(label)
    for i in range(len(props)):

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - search_radius, 0)):int(min(centroid[0] + search_radius, label.shape[0])),
                       int(max(centroid[1] - search_radius, 0)):int(min(centroid[1] + search_radius, label.shape[1]))
                       ]
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
        if np.max(nucleus_crop_dist) > 0 and not apply_clipping:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
        label_dist[
        int(max(centroid[0] - search_radius, 0)):int(min(centroid[0] + search_radius, label.shape[0])),
        int(max(centroid[1] - search_radius, 0)):int(min(centroid[1] + search_radius, label.shape[1]))
        ] += nucleus_crop_dist

    if apply_clipping:
        # Clip values
        label_dist = np.clip(label_dist, 0, clip_val)
        # Scale to [0, 1]
        label_dist = label_dist / clip_val

    return label_dist.astype(np.float32)


def distance_label(label, search_radius):
    """ Cell and neigbhor distance label creation (Euclidean distance).

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param search_radius: Defines the area to compute the distance transform (smaller radius in px decreases the computation time).
        :type search_radius: int
    :return: Cell distance label image, neighbor distance label image.
    """

    # Preallocation
    label_dist = np.zeros(shape=label.shape, dtype=np.float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=np.float)

    # Get Borders in-between touching cells
    label_border = (border_label(label) == 2)

    # Find centroids, crop image, calculate distance transforms
    props = measure.regionprops(label)
    for i in range(len(props)):

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - search_radius, 0)):int(min(centroid[0] + search_radius, label.shape[0])),
                       int(max(centroid[1] - search_radius, 0)):int(min(centroid[1] + search_radius, label.shape[1]))
                       ]
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
        max_dist = np.max(nucleus_crop_dist)
        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / max_dist
        else:
            continue
        label_dist[
        int(max(centroid[0] - search_radius, 0)):int(min(centroid[0] + search_radius, label.shape[0])),
        int(max(centroid[1] - search_radius, 0)):int(min(centroid[1] + search_radius, label.shape[1]))
        ] += nucleus_crop_dist

        # Get crop containing neighboring nuclei
        nucleus_neighbor_crop = np.copy(label[
                                int(max(centroid[0] - search_radius, 0)):int(
                                    min(centroid[0] + search_radius, label.shape[0])),
                                int(max(centroid[1] - search_radius, 0)):int(
                                    min(centroid[1] + search_radius, label.shape[1]))
                                ])

        # No need to calculate neighbor distances if no neighbor is in the crop
        if len(get_nucleus_ids(nucleus_neighbor_crop)) <= 1:
            continue

        # Convert background to nucleus id
        nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
        nucleus_neighbor_crop[nucleus_neighbor_crop == 0] = props[i].label
        nucleus_neighbor_crop[nucleus_neighbor_crop != props[i].label] = 0
        nucleus_neighbor_crop = nucleus_neighbor_crop > 0
        nucleus_neighbor_crop_dist = distance_transform_edt(nucleus_neighbor_crop)
        nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist * nucleus_neighbor_crop_nucleus
        if np.max(nucleus_neighbor_crop_dist) > 0:
            denominator = np.minimum(max_dist + 3,  # larger than max_dist since scaled later on (improves small cells)
                                     np.max(nucleus_neighbor_crop_dist))
            nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist / denominator
            nucleus_neighbor_crop_dist = np.clip(nucleus_neighbor_crop_dist, 0, 1)
        else:
            nucleus_neighbor_crop_dist = 1
        nucleus_neighbor_crop_dist = (1 - nucleus_neighbor_crop_dist) * nucleus_neighbor_crop_nucleus
        label_dist_neighbor[
        int(max(centroid[0] - search_radius, 0)):int(min(centroid[0] + search_radius, label.shape[0])),
        int(max(centroid[1] - search_radius, 0)):int(min(centroid[1] + search_radius, label.shape[1]))
        ] += nucleus_neighbor_crop_dist

    # Add neighbor distances in-between close but not touching cells with bottom-hat transform / fill gaps
    label_closed, label_closed_corr = bottom_hat_closing(label=label)
    props = measure.regionprops(label_closed)
    # Remove artifacts in the gap class
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    for obj_props in props:
        obj = (label_closed == obj_props.label)
        # There should be no high grayscale values around artifacts
        obj_boundary = ndimage.binary_dilation(obj, kernel) ^ obj
        if obj_props.area <= 20:
            th = 5
        elif obj_props.area <= 30:
            th = 8
        elif obj_props.area <= 50:
            th = 10
        else:
            th = 20
        if np.sum(obj_boundary * label_dist_neighbor) < th:  # Complete in background
            label_closed_corr[obj] = 0

    # label_dist_neighbor = np.maximum(label_dist_neighbor, label_gap.astype(label_dist_neighbor.dtype))
    label_dist_neighbor = np.maximum(label_dist_neighbor, label_closed_corr.astype(label_dist_neighbor.dtype))
    label_dist_neighbor = np.maximum(label_dist_neighbor, label_border.astype(label_dist_neighbor.dtype))

    # Scale neighbor distances
    label_dist_neighbor = 1 / np.sqrt(0.65 + 0.5 * np.exp(-11 * (label_dist_neighbor - 0.75))) - 0.19
    label_dist_neighbor = np.clip(label_dist_neighbor, 0, 1)
    label_dist_neighbor = grey_closing(label_dist_neighbor, size=(3, 3))

    return label_dist.astype(np.float32), label_dist_neighbor.astype(np.float32)
