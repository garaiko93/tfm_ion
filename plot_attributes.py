import matplotlib as mpl
mpl.use('agg')
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import tkinter
import seaborn as sb
from matplotlib import pyplot as plt

# functions from other scripts
from vioboxPlot import violinboxplot

# -----------------------------------------------------------------------------
# PLOT OF DICTIONARY SHAPE ATTRIBUTES
# -----------------------------------------------------------------------------
# DATA PREPARATION TO PLOT
def data_setup(study_areas, attr_name, plot_title, list_areas):
    if list_areas == 'All':
        areas = ['sion', 'linthal', 'frutigen', 'neuchatel', 'zermatt', 'locarno', 'luzern', 'bern',
                 'zurich_kreis', 'lausanne', 'lugano', 'stgallen']
    elif list_areas == 'Rural':
        areas = ['sion', 'linthal', 'frutigen', 'neuchatel', 'zermatt', 'locarno']
    elif list_areas == 'Urban':
        areas = ['luzern', 'bern', 'zurich_kreis', 'lausanne', 'lugano', 'stgallen']
    else:
        raise Exception('list_areas was not well defined (All, Urban or Rural)')

    data_to_plot = []
    x_axis = []
    for area in areas:
        study_area_dir = str(study_areas) + '/' + str(area)
        file = open(str(study_area_dir) + "/attr_" + str(attr_name) + ".pkl", 'rb')
        attr_dict = pickle.load(file)

        data = list(attr_dict.values())
        data_to_plot.append(data)
        x_axis.append(area)

    plt.figure(figsize=(16, len(data_to_plot)*1.5))

    if attr in ['edge_betweenness', 'node_betweenness']:
        logPercentile = 0.9
        outliers = True
        ax = plt.gca()
    elif attr in ['edge_load_centrality']:
        logPercentile = 0.9
        outliers = True
        ax = plt.gca()
        ax.set_xscale('log')
    elif attr in ['clustering']:
        logPercentile = None
        outliers = True
        ax = plt.gca()
        ax.set_xscale('log')
    else:
        logPercentile = None
        outliers = True
        ax = plt.gca()

    # ax = plt.gca()
    violinboxplot(data_to_plot, labels=x_axis, ax=ax, showModes=True, showCounts=True, outliers=outliers,
                  title=str(plot_title) + " - " + str(list_areas) + " Study Areas", logPercentile=logPercentile)
    plt.savefig(r'C:/Users/Ion/TFM/data/attribute_plots/' + str(list_areas) + '/' + str(attr_name) + '.png')
    print('Plot saved: ' + str(list_areas) + ' ' + str(attr_name))


attr_dict = {
    'avg_degree_connect': 'Average degree connectivity',
    'avg_neighbor_degree': 'Average Neighbour Degree',
    'clustering': 'Clustering Coefficient', #is not useful
    'degree_centrality': 'Degree of Centrality',
    'eccentricity': 'Eccentricity',
    'edge_betweenness': 'Edge Betweenness Centrality',
    'edge_load_centrality': 'Edge Load Centrality',
    'node_betweenness': 'Node Betweenness Centrality',
    'node_closeness_length': 'Node Closeness Centrality by Distance',
    'node_closeness_time': 'Node Closeness Centrality by Time',
    'node_load_centrality': 'Node Load Centrality',
    'node_straightness': 'Node Straightness Centrality'
}

# attr = 'clustering'
# env = 'Urban'
# data_setup(r'C:/Users/Ion/TFM/data/study_areas/', attr, attr_dict[attr], env)

for env in ['All', 'Rural', 'Urban']:
    for attr in list(attr_dict):
        data_setup('C:/Users/Ion/TFM/data/study_areas/', attr, attr_dict[attr], env)


# -----------------------------------------------------------------------------
# PLOT OF INTEGER ATTRIBUTES
# -----------------------------------------------------------------------------

study_area_dir = r'C:/Users/Ion/TFM/data/study_areas'
# attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area', dtype=object)
int_attr = ['network_distance',
            'population',
            'trips',
            'n_intersection',
            'node_d_km',
            'intersection_d_km',
            'edge_d_km',
            'circuity_avg',
            'edge_d_km']

rural = ['sion', 'linthal', 'frutigen', 'neuchatel', 'zermatt', 'locarno', 'plateau']
urban = ['luzern', 'bern', 'zurich_kreis', 'lausanne', 'lugano', 'stgallen']
all = rural + urban

rural_df = attr_df.loc[rural]
urban_df = attr_df.loc[urban]
all_df = attr_df.loc[all]

for df in [rural_df, urban_df, all_df]:
    for attr in int_attr:
        list_values = [int(d) for d in attr_df[attr].tolist()]
        labels = df.index

rural = ['sion', 'linthal', 'frutigen', 'neuchatel', 'zermatt', 'locarno', 'plateau']
list_values = [float(d) for d in rural_df['edge_d_km'].tolist()]

ax = plt.gca()
plt.plot(rural, list_values)
#
# import numpy as np
# x = np.arange(1,len(rural),1)
# y = np.array([20,21,22,23])
# my_xticks = ['John','Arnold','Mavis','Matt']
# plt.xticks(x, rural)
# plt.plot(x, list_values)
# plt.show()


# Pair wise plot
study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'

# df = sb.load_dataset('iris')
df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area', dtype=object)

for i in range(len(df)):
    # print(attr_df.index[i], i)
    for column in attr_df.columns:
        try:
            # FLOAT ATTRIBUTES
            str_val = df.iloc[i][column]
            flo_val = float(str_val)
            df.at[df.index[i], column] = flo_val
        except:
            pass
type(df.iloc[0]['network_distance'])

df2 = df[['network_distance', 'node_d_km','area_type', 'edge_d_km']]
df2 = df[['network_distance', 'node_d_km','area_type', 'edge_d_km']]
sb.set_style("ticks")
# sb.pairplot(df[['sepal_length', 'sepal_width', 'species']], hue='species', diag_kind="kde", kind="scatter", palette="husl")
sb.pairplot(df, hue='area_type', diag_kind="kde", kind="scatter", palette="husl")
plt.show()



# -----------------------------------------------------------------------------
# VIOBOXPLOT FUNCTION CALL
# -----------------------------------------------------------------------------
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(data_to_plot, labels=x_axis, showModes=True, showCounts=True,
#               title="All Study Areas - Node Straightness Centrality")


# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(df, x = 'values', y = ['col1', 'col2'],
#               showModes=True, ax = ax, logPercentile=0.9, labels=labels7, showCounts=True,
#              title="Dataframe grouped by col1 and col2")


# -----------------------------------------------------------------------------
# # box plot
# -----------------------------------------------------------------------------
# fig = plt.figure(1, figsize=(9, 6))
# # fig.suptitle('Rural Study Areas - Node Straightness Centrality', fontsize=14, fontweight='bold')
# fig.suptitle('All Study Areas - Node Straightness Centrality', fontsize=14, fontweight='bold')
# ax = fig.add_subplot(111)
# ax.set_xticklabels(x_axis)

# bp = ax.boxplot(data_to_plot)
# bp = ax.boxplot(data_to_plot, showfliers=False)

# # add patch_artist=True option to ax.boxplot()
# # to get fill color
# bp = ax.boxplot(data_to_plot, patch_artist=True)
#
# # change outline color, fill color and linewidth of the boxes
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
#
# # plt.show()
# plt.savefig("data/attribute_plots/all_areas_straightness.png")

# -----------------------------------------------------------------------------
# violin plot
# -----------------------------------------------------------------------------
# fig = plt.figure()
# ax = fig.add_subplot(111)
# bp = ax.violinplot(data_to_plot, showfliers=False)
# plt.show()




