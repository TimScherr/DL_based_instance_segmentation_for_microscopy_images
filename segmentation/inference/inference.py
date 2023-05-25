import gc
import json

import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F

from multiprocessing import cpu_count
from skimage.measure import regionprops, label
from skimage.transform import resize

from segmentation.inference.ctc_dataset import CTCDataSet, pre_processing_transforms
from segmentation.inference.postprocessing import *
from segmentation.inference.postprocessing_naylor import post_process
from segmentation.utils.unets import build_unet


def inference_2d_ctc(model, data_path, result_path, device, batchsize, cell_type, pparams, apply_merging=False, num_gpus=None):
    """ Inference function for 2D Cell Tracking Challenge data sets.

    :param model: Path to the model to use for inference.
        :type model: pathlib Path object.
    :param data_path: Path to the directory containing the Cell Tracking Challenge data sets.
        :type data_path: pathlib Path object
    :param result_path: Path to the results directory.
        :type result_path: pathlib Path object
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param batchsize: Batch size.
        :type batchsize: int
    :param num_gpus: Number of GPUs to use in GPU mode (enables larger batches)
        :type num_gpus: int
    :return: None
    """

    # Load model json file to get architecture + filters
    with open(model.parent / (model.stem + '.json')) as f:
        model_settings = json.load(f)

    # Build model
    ch_out = 3
    if model_settings['label_type'] in ['distance', 'cell_dist', 'dist']:
        ch_out = 1
    elif model_settings['label_type'] in ['j4', 'jcell']:
        ch_out = 4
    net = build_unet(unet_type=model_settings['architecture'][0],
                     act_fun=model_settings['architecture'][2],
                     pool_method=model_settings['architecture'][1],
                     normalization=model_settings['architecture'][3],
                     device=device,
                     num_gpus=num_gpus,
                     ch_in=1,
                     ch_out=ch_out,
                     filters=model_settings['architecture'][4])

    # Save raw predictions for frames with GTs
    path_gt = data_path.parent / f"{data_path.stem}_GT" / 'SEG'
    gt_ids = sorted(path_gt.glob('*.tif'))
    save_ids = []
    for gt_idx in gt_ids:
        save_ids.append(gt_idx.stem.split('man_seg')[-1])

    # Get number of GPUs to use and load weights
    if not num_gpus:
        num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        net.module.load_state_dict(torch.load(str(model), map_location=device))
    else:
        net.load_state_dict(torch.load(str(model), map_location=device))
    net.eval()
    torch.set_grad_enabled(False)

    # Get images to predict
    ctc_dataset = CTCDataSet(data_dir=data_path,
                             transform=pre_processing_transforms(apply_clahe=False, scale_factor=1))
    if device.type == "cpu":
        num_workers = 0
    else:
        try:
            num_workers = cpu_count() // 2
        except AttributeError:
            num_workers = 4
    if num_workers <= 2:  # Probably Google Colab --> use 0
        num_workers = 0
    num_workers = np.minimum(num_workers, 16)
    dataloader = torch.utils.data.DataLoader(ctc_dataset, batch_size=batchsize, shuffle=False, pin_memory=True,
                                             num_workers=num_workers)

    # Predict images (iterate over images/files)
    for sample in dataloader:

        img_batch, ids_batch, pad_batch, img_size = sample
        img_batch = img_batch.to(device)

        if batchsize > 1:  # all images in a batch have same dimensions and pads
            pad_batch = [pad_batch[i][0] for i in range(len(pad_batch))]
            img_size = [img_size[i][0] for i in range(len(img_size))]

        # Prediction
        if model_settings['label_type'] in ['distance', 'adapted_border']:
            prediction_border_batch, prediction_cell_batch = net(img_batch)
            if model_settings['label_type'] == 'adapted_border':
                prediction_border_batch = F.softmax(prediction_border_batch, dim=1)
                prediction_cell_batch = torch.sigmoid(prediction_cell_batch)
                prediction_cell_batch = prediction_cell_batch[:, 0, pad_batch[0]:, pad_batch[1]:].cpu().numpy()
                prediction_border_batch = prediction_border_batch[:, :, pad_batch[0]:, pad_batch[1]:].cpu().numpy()
                prediction_border_batch = np.transpose(prediction_border_batch, (0, 2, 3, 1))
            else:
                prediction_cell_batch = prediction_cell_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
                prediction_border_batch = prediction_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
        else:
            prediction_batch = net(img_batch)
            if model_settings['label_type'] in ['boundary', 'border', 'j4']:
                prediction_batch = F.softmax(prediction_batch, dim=1)
            prediction_batch = prediction_batch[:, :, pad_batch[0]:, pad_batch[1]:].cpu().numpy()
            prediction_batch = np.transpose(prediction_batch, (0, 2, 3, 1))

        # Go through predicted batch and apply post-processing (not parallelized)
        for h in range(len(img_batch)):

            print('         ... processing {0} ...'.format(ids_batch[h]))

            file_id = ids_batch[h].split('t')[-1] + '.tif'

            if model_settings['label_type'] == 'distance':
                prediction_instance = distance_postprocessing(border_prediction=prediction_border_batch[h],
                                                              cell_prediction=prediction_cell_batch[h],
                                                              th_seed=pparams[0],
                                                              th_cell=pparams[1],
                                                              apply_merging=apply_merging)
            elif model_settings['label_type'] == 'dist':
                prediction_instance = post_process(prob_image=np.squeeze(prediction_batch[h]),
                                                   param=pparams[0],
                                                   thresh=pparams[1])
            elif model_settings['label_type'] == 'cell_dist':
                prediction_instance = cell_dist_postprocessing(cell_prediction=prediction_batch[h],
                                                               th_seed=pparams[0],
                                                               th_cell=pparams[1])
            elif model_settings['label_type'] == 'boundary':
                prediction_instance = boundary_postprocessing(prediction=prediction_batch[h])
            elif model_settings['label_type'] == 'border':
                prediction_instance = border_postprocessing(prediction=prediction_batch[h])
            elif model_settings['label_type'] == 'adapted_border':
                prediction_instance = adapted_border_postprocessing(border_prediction=prediction_border_batch[h],
                                                                    cell_prediction=prediction_cell_batch[h])
            elif model_settings['label_type'] == 'j4':
                prediction_instance = j4_postprocessing(prediction=prediction_batch[h])

            prediction_instance = foi_correction(mask=prediction_instance, cell_type=cell_type)

            # Save raw predictions for frames with GT available
            tiff.imsave(str(result_path / ('mask' + file_id)), prediction_instance, compress=1)
            if ids_batch[h].split('t')[-1] in save_ids:  # Save raw predictions
                if model_settings['label_type'] == 'distance':
                    tiff.imsave(str(result_path / ('raw_cell' + file_id)), prediction_cell_batch[h, ..., 0].astype(np.float32), compress=1)
                    tiff.imsave(str(result_path / ('raw_border' + file_id)), prediction_border_batch[h, ..., 0].astype(np.float32), compress=1)
                elif model_settings['label_type'] == 'adapted_border':
                    tiff.imsave(str(result_path / ('raw_cell' + file_id)), prediction_cell_batch[h].astype(np.float32), compress=1)
                    tiff.imsave(str(result_path / ('raw_border' + file_id)), prediction_border_batch[h].astype(np.float32), compress=1)
                else:
                    tiff.imsave(str(result_path / ('raw_' + file_id)), np.squeeze(prediction_batch[h]).astype(np.float32), compress=1)

    # Clear memory
    del net
    gc.collect()

    return None
