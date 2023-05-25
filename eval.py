import json
import numpy as np
import pandas as pd
import tifffile as tiff
import torch

from pathlib import Path
from skimage import measure

from segmentation.inference.inference import inference_2d_ctc
from segmentation.utils.metrics import count_det_errors, ctc_metrics
from segmentation.utils import utils


def main():

    # Paths
    path_data = Path(__file__).parent / 'training_data'
    path_models = Path(__file__).parent / 'models' / 'all'
    path_best_models = Path(__file__).parent / 'models' / 'best'
    path_ctc_metric = Path(__file__).parent / 'evaluation_software'

    cell_types = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "Fluo-N2DL-HeLa", "PhC-C2DL-PSC"]
    eval_sets = ["02"]

    # Get number of cells for each data set and normalize corresponding columns
    cell_counts = {"BF-C2DL-HSC": 0,
                   "BF-C2DL-MuSC": 0,
                   "Fluo-N2DL-HeLa": 0,
                   "PhC-C2DL-PSC": 0}
    for cell_type in cell_types:
        counts = 0
        file_ids = sorted((path_data / cell_type / "02_GT" / "TRA").glob('*.tif'))
        for file_idx in file_ids:
            tra_gt = tiff.imread(str(file_idx))
            tra_gt = measure.label(tra_gt, background=0)
            counts += np.max(tra_gt)
        cell_counts[cell_type] = counts
    print(cell_counts)

    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0

    # Check if dataset consists in training_data folder
    es = 0
    for cell_type in cell_types:
        if not (path_data / cell_type).exists():
            print('No data for cell type "{}" found in {}'.format(cell_type, path_data))
            es = 1
    if es == 1:
        return

    # Check if evaluation metric is available
    if not path_ctc_metric.is_dir():
        raise Exception('No evaluation software found. Run the skript download_data.py')

    # Get models and cell types to evaluate
    models = sorted(path_models.glob("*.pth"))
    if len(models) == 0:
        raise Exception('No models to evaluate found.')

    # Go through model list and evaluate for stated cell_types
    scores = []
    for model in models:
        # Load model json file to get architecture + filters
        with open(model.parent / (model.stem + '.json')) as f:
            model_settings = json.load(f)

        merging_set = [False]
        if model_settings['label_type'] in ['distance', 'cell_dist']:
            param_list = [[0.4, 0.08],
                          [0.5, 0.08]
                          ]
            if model_settings['label_type'] == 'distance':
                merging_set = [False, True]
        elif model_settings['label_type'] == 'dist':
            # param_list = [[0, 0.5],
            #               [1, 0.5]
            #               ]
            param_list = [#[0, 0.5],
                          [1, 0.5]
                          ]
        else:
            param_list = [[0, 0]]

        for params in param_list:

            for merging in merging_set:

                for ct in cell_types:

                    if merging and ct in ['BF-C2DL-HSC', 'Fluo-N2DL-HeLa', 'PhC-C2DL-PSC']:
                        continue

                    results = {'cell type': ct,
                               'model': model.stem,
                               'method': model_settings['label_type'],
                               'optimizer': model_settings['optimizer'],
                               'param 0': params[0],
                               'param 1': params[1],
                               'merging': merging,
                               'dataset': model.stem.split('_')[0],
                               # 'DET (01)': np.nan,
                               # 'DET (02)': np.nan,
                               'DET': np.nan,
                               # 'SEG (01)': np.nan,
                               # 'SEG (02)': np.nan,
                               'SEG': np.nan,
                               # 'OP_CSB (01)': np.nan,
                               # 'OP_CSB (02)': np.nan,
                               'OP_CSB': np.nan,
                               # 'SO (01)': np.nan,
                               # 'SO (02)': np.nan,
                               'SO': np.nan,
                               # 'FPV (01)': np.nan,
                               # 'FPV (02)': np.nan,
                               'FPV': np.nan,
                               # 'FNV (01)': np.nan,
                               # 'FNV (02)': np.nan,
                               'FNV': np.nan,
                               }
                    for eval_set in eval_sets:
                        print(f'Evaluate {model.stem}')
                        if merging:
                            path_seg_results = path_data / ct / f"{ct}_RES_{eval_set}_{model.stem}_{params[0]}_{params[1]}_merging"
                        else:
                            path_seg_results = path_data / ct / f"{ct}_RES_{eval_set}_{model.stem}_{params[0]}_{params[1]}"
                        path_seg_results.mkdir(exist_ok=True)

                        # Load existing results
                        if (path_seg_results / "DET_log.txt").exists() and (path_seg_results / "SEG_log.txt").exists():
                            det_measure, so, fnv, fpv = count_det_errors(path_seg_results / "DET_log.txt")
                            seg_measure = utils.get_seg_score(path_seg_results / "SEG_log.txt")
                        else:
                            if '2D' in ct:
                                inference_2d_ctc(model=model,
                                                 data_path=path_data / ct / eval_set,
                                                 result_path=path_seg_results,
                                                 device=device,
                                                 batchsize=8,
                                                 num_gpus=num_gpus,
                                                 pparams=params,
                                                 apply_merging=merging,
                                                 cell_type=ct)

                            seg_measure, det_measure = ctc_metrics(path_data=path_data / ct,
                                                                   path_results=path_seg_results,
                                                                   path_software=path_ctc_metric,
                                                                   subset=eval_set,
                                                                   mode='GT')

                            # For evaluation on silver truth only the SEG measure is used/calculated
                            _, so, fnv, fpv = count_det_errors(path_seg_results / "DET_log.txt")

                        results[f'DET'] = det_measure
                        results[f'SEG'] = seg_measure
                        results[f'OP_CSB'] = np.nansum([det_measure, seg_measure]) / 2
                        results[f'SO'] = so / cell_counts[cell_type]
                        results[f'FPV'] = fpv / cell_counts[cell_type]
                        results[f'FNV'] = fnv / cell_counts[cell_type]

                    # results['DET'] = np.nansum([results['DET (01)'], results['DET (02)']]) / 2
                    # results['SEG'] = np.nansum([results['SEG (01)'], results['SEG (02)']]) / 2
                    # results['OP_CSB'] = np.nansum([results['OP_CSB (01)'], results['OP_CSB (02)']]) / 2
                    # results['SO'] = np.nansum([results['SO (01)'], results['SO (02)']])
                    # results['FPV'] = np.nansum([results['FPV (01)'], results['FPV (02)']])
                    # results['FNV'] = np.nansum([results['FNV (01)'], results['FNV (02)']])
                    scores.append(results)

                    scores_df = pd.DataFrame(scores)
                    scores_df.to_csv(path_best_models.parent / "metrics.csv", header=True, index=False)

    # Convert to dataframe, merge with existing results and save
    scores_df = pd.DataFrame(scores)
    # if (path_best_models.parent / "metrics.csv").is_file():
    #     old_scores_df = pd.read_csv(path_best_models.parent / "metrics.csv")
    #     scores_df = pd.concat([scores_df, old_scores_df])
    #     # Delete duplicate entries
    #     scores_df = scores_df.drop_duplicates(subset=['model', 'method', 'cell type', 'optimizer'], keep='first')
    scores_df = scores_df.sort_values(by=['cell type', 'model'])
    scores_df.to_csv(path_best_models.parent / "metrics.csv", header=True, index=False)


if __name__ == "__main__":

    main()
