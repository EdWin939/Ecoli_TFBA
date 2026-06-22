from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


experiments_path= "/Users/eduardmrug/Documents/Honours research/Labwork/Experiments/"
experiments=['20250729_GWT305','20250729_GKO305','20250729_MWT305/', '20250729_MKO305/']
def violinis(experiments, type):
    plot_data = []
    for exp in experiments:
        path_base = os.path.join(experiments_path, exp)
        
        path_out = os.path.join(path_base, "Analysis")
        
        df_total = pd.read_pickle(os.path.join(path_out, "dataframe_fluorescence_bandpass-values.pkl"))
        if type=='max':
             plot_data.append(df_total[df_total["ratio_max"] <= 1]['ratio_max'])
        else:
            plot_data.append(df_total[df_total["ratio_mean"] <= 1]['ratio_mean'])
    # Plotting
    plt.figure(figsize=(8, 4))
    plt.violinplot(plot_data, quantiles=[[0.25, 0.5, 0.75] for i in experiments])
    plt.title(f'{type} CFP/YFP ratio')
    plt.xlabel('Experiment')
    plt.ylabel(f'{type} CFP/YFP')
    plt.xticks(ticks=np.arange(1, len(experiments) + 1), labels=experiments)
    plt.grid(True)
    plt.savefig(os.path.join(experiments_path, f'violin_plot_{type}.png'))
    plt.show()

violinis(experiments, 'max')