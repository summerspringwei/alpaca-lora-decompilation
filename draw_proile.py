##########################
### MODEL
##########################

from torch import nn
###### NEW ####################################################
from torch.utils.checkpoint import checkpoint_sequential
###############################################################
import numpy as np
import pickle
import matplotlib.pyplot as plt


def draw_latency_curve(file_path: str):
    profile_data = pickle.load(open(file_path, 'rb'))
    time_list = profile_data["time_list"]
    cur_len_list = profile_data["cur_len_list"][:-1]
    first_list = time_list[:-1]
    second_list = time_list[1:]
    latency_list = (np.array(second_list) - np.array(first_list))/ 1e6
    for latency in latency_list:
        print(latency)
    print()

    plt.plot(cur_len_list, latency_list)
    plt.savefig(f"{file_path}-latency_plot.png")
    plt.clf()

draw_latency_curve('latency_2024-09-05_13:06:32.pkl')
draw_latency_curve('latency_2024-09-05_13:18:05.pkl')
