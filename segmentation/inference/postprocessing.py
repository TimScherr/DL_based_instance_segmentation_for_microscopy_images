import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.segmentation import watershed
from skimage import measure

from segmentation.utils.utils import get_nucleus_ids


def foi_correction(mask, cell_type):
    """ Field of interest correction for Cell Tracking Challenge data (see
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf and
    https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf )

    :param mask: Segmented cells.
        :type mask:
    :param cell_type: Cell Type.
        :type cell_type: str
    :return: FOI corrected segmented cells.
    """

    if cell_type in ['DIC-C2DH-HeLa', 'Fluo-C2DL-Huh7', 'Fluo-C2DL-MSC', 'Fluo-C3DH-H157', 'Fluo-N2DH-GOWT1',
                     'Fluo-N3DH-CE', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373']:
        E = 50
    elif cell_type in ['BF-C2DL-HSC', 'BF-C2DL-MuSC', 'Fluo-C3DL-MDA231', 'Fluo-N2DL-HeLa', 'PhC-C2DL-PSC']:
        E = 25
    else:
        E = 0

    if len(mask.shape) == 2:
        foi = mask[E:mask.shape[0] - E, E:mask.shape[1] - E]
    else:
        foi = mask[:, E:mask.shape[1] - E, E:mask.shape[2] - E]

    ids_foi = get_nucleus_ids(foi)
    ids_prediction = get_nucleus_ids(mask)
    for id_prediction in ids_prediction:
        if id_prediction not in ids_foi:
            mask[mask == id_prediction] = 0

    return mask


def distance_postprocessing(border_prediction, cell_prediction, th_seed, th_cell, apply_merging=False):
    """ Post-processing for distance label (cell + neighbor) prediction.

    :param border_prediction: Neighbor distance prediction.
    :type border_prediction:
    :param cell_prediction: Cell distance prediction.
    :type cell_prediction:
    :param th_seed: Threshold for seed/marker extraction
    :type th_seed: float
    :param th_cell: Threshold for cell size
    :type th_cell: float
    :return: Instance segmentation mask.
    """

    # Smooth predictions slightly + clip border prediction (to avoid negative values being positive after squaring)
    sigma_cell = 0.5
    # sigma_border = 0.5

    cell_prediction = gaussian_filter(cell_prediction, sigma=sigma_cell)
    # border_prediction = gaussian_filter(border_prediction, sigma=sigma_border)
    border_prediction = np.clip(border_prediction, 0, 1)

    # Get mask for watershed
    mask = cell_prediction > th_cell

    # Get seeds for marker-based watershed
    borders = np.tan(border_prediction ** 2)
    borders[borders < 0.05] = 0
    borders = np.clip(borders, 0, 1)
    cell_prediction_cleaned = (cell_prediction - borders)
    seeds = cell_prediction_cleaned > th_seed
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    areas = []
    for i in range(len(props)):
        areas.append(props[i].area)
    if len(areas) > 0:
        min_area = 0.10 * np.mean(np.array(areas))
    else:
        min_area = 0
    min_area = np.maximum(min_area, 4)

    for i in range(len(props)):
        if props[i].area <= min_area:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)

    if apply_merging:
        # Get borders between touching cells
        label_bin = prediction_instance > 0
        pred_boundaries = cv2.Canny(prediction_instance.astype(np.uint8), 1, 1) > 0
        pred_borders = cv2.Canny(label_bin.astype(np.uint8), 1, 1) > 0
        pred_borders = pred_boundaries ^ pred_borders
        pred_borders = measure.label(pred_borders)
        for border_id in get_nucleus_ids(pred_borders):
            pred_border = (pred_borders == border_id)
            if np.sum(border_prediction[pred_border]) / np.sum(pred_border) < 0.075:  # likely splitted due to shape
                # Get ids to merge
                pred_border_dilated = binary_dilation(pred_border, np.ones(shape=(3, 3), dtype=np.uint8))
                merge_ids = get_nucleus_ids(prediction_instance[pred_border_dilated])
                if len(merge_ids) == 2:
                    prediction_instance[prediction_instance == merge_ids[1]] = merge_ids[0]
        prediction_instance = measure.label(prediction_instance)

    return np.squeeze(prediction_instance.astype(np.uint16))


def cell_dist_postprocessing(cell_prediction, th_seed, th_cell):
    """ Post-processing for distance label (cell + neighbor) prediction.
    :param cell_prediction: Cell distance prediction.
    :type cell_prediction:
    :param th_seed: Threshold for seed/marker extraction
    :type th_seed: float
    :param th_cell: Threshold for cell size
    :type th_cell: float
    :return: Instance segmentation mask.
    """

    # Smooth predictions slightly + clip border prediction (to avoid negative values being positive after squaring)
    sigma_cell = 0.5
    cell_prediction = gaussian_filter(cell_prediction, sigma=sigma_cell)

    # Get mask for watershed
    mask = cell_prediction > th_cell

    # Get seeds for marker-based watershed
    seeds = cell_prediction > th_seed
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    # areas = []
    # for i in range(len(props)):
    #     areas.append(props[i].area)
    # if len(areas) > 0:
    #     min_area = 0.10 * np.mean(np.array(areas))
    # else:
    #     min_area = 0
    # min_area = np.maximum(min_area, 4)

    for i in range(len(props)):
        if props[i].area <= 4:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)

    return np.squeeze(prediction_instance.astype(np.uint16))


def boundary_postprocessing(prediction):
    """ Post-processing for boundary label prediction.

    :param prediction: Boundary label prediction.
    :type prediction:
    :return: Instance segmentation mask, binary raw prediction (0: background, 1: cell, 2: boundary).
    """

    # Binarize the channels
    prediction_bin = np.argmax(prediction, axis=-1).astype(np.uint16)

    # Get mask to flood with watershed
    mask = (prediction_bin == 1)  # only interior cell class belongs to cells

    # Get seeds for marker-based watershed
    seeds = (prediction[:, :, 1] * (1 - prediction[:, :, 2])) > 0.5
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 4:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    return np.squeeze(prediction_instance.astype(np.uint16))


def border_postprocessing(prediction):
    """ Post-processing for border label prediction.

    :param prediction: Border label prediction.
        :type prediction:
    :return: Instance segmentation mask, binary raw prediction (0: background, 1: cell, 2: border).
    """

    # Binarize the channels
    prediction_bin = np.argmax(prediction, axis=-1).astype(np.uint16)

    # Get mask to flood with watershed
    mask = (prediction_bin > 0)  # border class belongs to cells

    # Get seeds for marker-based watershed
    seeds = (prediction[:, :, 1] * (1 - prediction[:, :, 2])) > 0.5
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 4:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    return np.squeeze(prediction_instance.astype(np.uint16))


def adapted_border_postprocessing(border_prediction, cell_prediction):
    """ Post-processing for adapted border label prediction.

    :param border_prediction: Adapted border prediction (3 channels).
        :type border_prediction:
    :param cell_prediction: Cell prediction (1 channel).
        :type cell_prediction:
    :return: Instance segmentation mask, binary border prediction (0: background, 1: cell, 2: border).
    """

    # Get mask to flood with watershed
    mask = cell_prediction > 0.5

    # Get seeds for marker-based watershed
    seeds = border_prediction[:, :, 1] * (1 - border_prediction[:, :, 2]) > 0.5
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 4:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    return np.squeeze(prediction_instance.astype(np.uint16))


def j4_postprocessing(prediction):
    """ Post-processing for j4 label prediction (background, cell, touching, gap).

    :param prediction: pena label prediction.
        :type prediction:
    :return: Instance segmentation mask, binary raw prediction (0: background, 1: cell, 2: border, 3: gap).
    """

    # Binarize the channels
    prediction_bin = np.argmax(prediction, axis=-1).astype(np.uint16)

    # Get mask to flood with watershed
    mask = (prediction_bin == 1) | (prediction_bin == 2)  # gap belongs to background

    # Get seeds for marker-based watershed
    # seeds = prediction_bin == 1  ## results in merged objects
    seeds = (prediction[:, :, 1] * (1 - prediction[:, :, 2]) * (1 - prediction[:, :, 3])) > 0.5
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 4:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    return np.squeeze(prediction_instance.astype(np.uint16))
