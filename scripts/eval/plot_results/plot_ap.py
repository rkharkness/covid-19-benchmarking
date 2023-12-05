import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

bar1 = [29.1, 8.4]
bar2 = [45.3, 14.2]
labels = ['AP','PA']

x = np.arange(len(labels)) # the label locations
width = 0.35  # the width of the bars
with plt.style.context(['science', 'nature']):
    fig, ax = plt.subplots()

    rects1 = ax.bar(x-width/2, bar1, width, label='Full CXR')
    rects2 = ax.bar(x+width/2, bar2, width, label='ROI CXR')
    ax.set_ylabel('Frequency of False Positives (\%)')
    ax.set_ylim(0,110)
    ax.set_xlabel('Projection')


    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/xcept_seg_ap.pdf")

