import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib

from pathlib import Path
from scipy.stats import gaussian_kde


def main():
    # Paths
    path_plots = Path(__file__).parent / 'plots'
    path_datasets = Path(__file__).parent / 'training_data'

    cell_types = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "Fluo-N2DL-HeLa", "PhC-C2DL-PSC"]
    optimizers = ["adam", "ranger"]
    methods = ["adapted_border", "boundary", "dist", "distance", "j4"]
    metrics = ["DET", "SEG", "OP_CSB", "FNV", "FPV", "SO"]

    scores_df = pd.read_csv(Path(__file__).parent / 'models' / "metrics.csv")

    scores_df = scores_df.drop(scores_df[scores_df['method'] == 'cell_dist'].index)
    scores_df = scores_df.drop(scores_df[scores_df['method'] == 'border'].index)

    scores_distance_df = scores_df[scores_df['method'] == 'distance']
    scores_df = scores_df.drop(scores_df[scores_df['merging'] == True].index)

    for optimizer in optimizers:

        # Drop duplicates: keep='first' (th_seed=0.4 for cell_dist and distance)
        scores_1_df = scores_df.drop_duplicates(subset=['cell type', 'model', 'optimizer'], keep='first')
        # Drop duplicates: keep='last' (th_seed=0.5 for cell_dist and distance)
        scores_2_df = scores_df.drop_duplicates(subset=['cell type', 'model', 'optimizer'], keep='last')
        # Extract scores for selected optimizer
        scores_1_df = scores_1_df.drop(scores_1_df[~(scores_1_df['optimizer'] == optimizer)].index)
        scores_2_df = scores_2_df.drop(scores_2_df[~(scores_2_df['optimizer'] == optimizer)].index)
        scores_distance_opt_df = scores_distance_df.drop(scores_distance_df[~(scores_distance_df['optimizer'] == optimizer)].index)

        # Plots
        for metric in metrics:
            plt.figure()
            ax = sns.boxplot(x='method', y=metric, hue="cell type", data=scores_2_df)  # pastel, dark, deep
            ax = sns.swarmplot(x='method', y=metric, hue="cell type", dodge=True, data=scores_2_df, size=2)

            ax.set_xlabel('')
            # [ax.axvline(x, color='w', linestyle='-') for x in np.arange(0.5, len(methods) - 1, 1)]
            handles, labels = ax.get_legend_handles_labels()
            if metric in ['SEG', 'DET', 'OP_CSB']:
                plt.legend(handles[0:4], labels[0:4], loc='lower right')
            else:
                plt.legend(handles[0:4], labels[0:4], loc='upper right')
            # plt.xticks(ax.get_xticks(), rotation=45)
            plt.savefig(str(path_plots / f'{metric}_{optimizer}.pdf'), bbox_inches='tight', dpi=300)
            tikzplotlib.save(str(path_plots / f"{metric}_{optimizer}.tex"),
                             axis_width=r'\axiswidth',
                             axis_height=r'\axisheight')
            plt.close()
            for method in methods:
                for cell_type in cell_types:
                    df = scores_2_df[scores_2_df['cell type'] == cell_type]
                    df = df[df['method'] == method]
                    df = df[metric]
                    if metric in ['SO', 'FPV', 'FNV']:  # in %
                        df = df * 100
                    df.to_csv(path_plots / f"{method}_{metric}_{cell_type}_{optimizer}.dat", header=True, index=False)

                    if method == 'distance':
                        for th in [0.4, 0.5]:
                            df = scores_distance_opt_df[scores_distance_opt_df['cell type'] == cell_type]
                            df = df.drop(df[~(df['param 0'] == th)].index)
                            for merging in [True, False]:
                                df2 = df.drop(df[~(df['merging'] == merging)].index)
                                df2 = df2[metric]
                                if len(df2) > 0:
                                    if metric in ['SO', 'FPV', 'FNV']:  # in %
                                        df2 = df2 * 100
                                    df2.to_csv(path_plots / f"{method}_{metric}_{cell_type}_{optimizer}_{th}_{merging}.dat",
                                               header=True, index=False)


if __name__ == "__main__":
    main()
