import matplotlib as mpl
import pandas as pd
import ast
import numpy as np
mpl.use('agg')
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import tkinter

data_to_plot = []
x_axis = []
# for area in ['luzern', 'bern', 'zurich_kreis', 'lausanne', 'lugano', 'stgallen']:
# for area in ['sion', 'linthal', 'frutigen', 'neuchatel', 'zermatt', 'locarno']:
for area in ['sion', 'linthal', 'frutigen', 'neuchatel', 'zermatt', 'locarno', 'luzern', 'bern',
             'zurich_kreis', 'lausanne', 'lugano', 'stgallen']:
    study_area_dir= r"C:/Users/Ion/TFM/data/study_areas/" + str(area)
    file = open(str(study_area_dir) + "/attr_node_straightness.pkl", 'rb')
    nodes_betw = pickle.load(file)

    data = list(nodes_betw.values())
    data_to_plot.append(data)
    x_axis.append(area)

# # box plot
fig = plt.figure(1, figsize=(9, 6))
# fig.suptitle('Rural Study Areas - Node Straightness Centrality', fontsize=14, fontweight='bold')
fig.suptitle('All Study Areas - Node Straightness Centrality', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.set_xticklabels(x_axis)

# bp = ax.boxplot(data_to_plot)
# bp = ax.boxplot(data_to_plot, showfliers=False)

# add patch_artist=True option to ax.boxplot()
# to get fill color
bp = ax.boxplot(data_to_plot, patch_artist=True)

# change outline color, fill color and linewidth of the boxes
# for box in bp['boxes']:
#     # change outline color
#     box.set( color='#7570b3', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#1b9e77' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp['whiskers']:
#     whisker.set(color='#7570b3', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp['caps']:
#     cap.set(color='#7570b3', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp['medians']:
#     median.set(color='#b2df8a', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp['fliers']:
#     flier.set(marker='o', color='#e7298a', alpha=0.5)

# plt.show()
plt.savefig("data/attribute_plots/all_areas_straightness.png")








# violin plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# bp = ax.violinplot(data_to_plot, showfliers=False)
# plt.show()




