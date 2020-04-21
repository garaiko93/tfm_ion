import matplotlib as mpl
mpl.use('agg') #only save
# mpl.use('TkAgg') #see plot
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import tkinter
import seaborn as sb
from matplotlib import pyplot as plt
import os
import numpy as np

# functions from other scripts
from vioboxPlot import violinboxplot

def save_plot(fig, path, fig_name):
    i = 0
    fig_path = str(path) + '/' + str(fig_name) + str(i) + '.png'
    while os.path.isfile(fig_path):
        i += 1
        fig_path = str(path) + '/' + str(fig_name) + str(i) + '.png'
    fig.savefig(fig_path)
    print('Plot saved: ' + str(fig_name))


# -----------------------------------------------------------------------------
# PLOT OF DICTIONARY SHAPE ATTRIBUTES
# -----------------------------------------------------------------------------
# DATA PREPARATION TO PLOT
def data_setup(study_areas, attr, plot_title, list_areas):
    if list_areas == 'All':
        areas = ['zermatt', 'locarno', 'chur', 'sion', 'linthal', 'frutigen', 'freiburg', 'neuchatel', 'plateau', 'luzern', 'bern',
                 'zurich_kreis', 'lausanne', 'lugano', 'stgallen']
    elif list_areas == 'Rural':
        areas = ['freiburg', 'neuchatel', 'plateau']
    elif list_areas == 'Urban':
        areas = ['luzern', 'bern', 'zurich_kreis', 'lausanne', 'lugano', 'stgallen']
    elif list_areas == 'Mountain':
        areas = ['zermatt', 'locarno', 'chur', 'sion', 'linthal', 'frutigen']
    else:
        raise Exception('list_areas was not well defined (All, Urban or Rural)')

    data_to_plot = []
    x_axis = []
    for area in areas:
        study_area_dir = str(study_areas) + '/' + str(area)
        file = open(str(study_area_dir) + "/attr_" + str(attr) + ".pkl", 'rb')
        attr_dict = pickle.load(file)

        data = list(attr_dict.values())
        data_to_plot.append(data)
        x_axis.append(area)

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(16, len(data_to_plot)*1.5))
    # plt.figure(figsize=(10, len(data_to_plot)*1))

    if attr in ['edge_betweenness',
                'node_betweenness',
                'btw_home_trip_production',
                'btw_empl_trip_generation',
                'btw_acc_trip_generation',
                'btw_acc_trip_production']:
        logPercentile = 0.9
        outliers = True
        ax = plt.gca()
    # elif attr in ['edge_load_centrality']:
    #     logPercentile = 0.9
    #     outliers = True
    #     ax = plt.gca()
    #     ax.set_xscale('log')
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
    # violinboxplot(data_to_plot, labels=x_axis, ax=ax, showModes=True, showCounts=True, outliers=outliers,
    #               title=str(plot_title) + " - " + str(list_areas) + " Study Areas", logPercentile=logPercentile)

    violinboxplot(data_to_plot, labels=x_axis, ax=ax, showModes=True, showCounts=True, outliers=outliers, logPercentile=logPercentile)
    plt.savefig(r'C:/Users/Ion/TFM/data/plots/attribute_plots/' + str(list_areas) + '/' + str(attr) + '.png')
    # plt.savefig(r'C:/Users/Ion/TFM/' + str(attr_name) + '.png')
    print('Plot saved: ' + str(list_areas) + ' ' + str(attr))


attr_dict = {
    'avg_degree_connect': 'Average Degree Connectivity',
    'avg_neighbor_degree': 'Average Neighbour Degree',
    'clustering': 'Clustering Coefficient', #is not useful
    'degree_centrality': 'Degree of Centrality',
    'eccentricity': 'Eccentricity',
    'edge_betweenness': 'Edge Betweenness Centrality',
    # 'edge_load_centrality': 'Edge Load Centrality',
    'node_betweenness': 'Node Betweenness Centrality',
    'node_closeness_length': 'Node Closeness Centrality by Distance',
    'node_closeness_time': 'Node Closeness Centrality by Time',
    # 'node_load_centrality': 'Node Load Centrality',
    'node_straightness': 'Node Straightness Centrality',
    'btw_home_trip_production': 'Betweenness-Accessibility: Links trip production potential',
    'btw_empl_trip_generation': 'Betweenness-Accessibility: Links trip generation potential',
    'btw_acc_trip_generation': 'Betweenness-Accessibility: Links accessibility to population',
    'btw_acc_trip_production': 'Betweenness-Accessibility: Links accessibility to employment',
}

# attr = 'btw_acc_trip_production'
# env = 'All'
# data_setup(r'C:/Users/Ion/TFM/data/study_areas/', attr, attr_dict[attr], env)
#
# for env in ['All', 'Rural', 'Urban', 'Mountain']:
#     for attr in list(attr_dict):
#         data_setup('C:/Users/Ion/TFM/data/study_areas/', attr, attr_dict[attr], env)


# -----------------------------------------------------------------------------
# GRID PLOT OF ATTRIBUTES
# -----------------------------------------------------------------------------

# https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
# https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
def grid_plot(df, fig_name):
    # Function to calculate correlation coefficient between two arrays
    def corr(x, y, **kwargs):
        # Calculate the value
        coef = np.corrcoef(x, y)[0][1]
        # Make the label
        label = r'$\rho$ = ' + str(round(coef, 2))
        # label = r'$\rho$ = ' + str("{:.3f}".format(coef))

        # Add the label to the plot
        ax = plt.gca()
        ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)

    # Create a pair grid instance
    grid = sb.PairGrid(data=df, hue='area_type', palette="husl")
    # grid = sb.PairGrid(data=df, hue='area_type', palette="husl", height=4, aspect=1)
    # grid = sb.PairGrid(data=df)

    # Map the plots to the locations
    grid = grid.map_upper(plt.scatter)
    grid = grid.map_upper(corr, corre=True)
    grid = grid.map_lower(sb.kdeplot)
    grid = grid.map_diag(plt.hist, bins=5, edgecolor='k')
    # grid = grid.map_diag(sb.distplot, hist=False)
    return grid


def facet_plot(df):
    for i in range(len(df)):
        for column in df.columns:
            try:
                # FLOAT ATTRIBUTES
                str_val = df.iloc[i][column]
                flo_val = float(str_val)
                df.at[df.index[i], column] = flo_val
            except:
                pass

    # plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 13})
    sb.set_style("ticks")
    # sb.pairplot(df2, hue='area_type', diag_kind="kde", kind="scatter", palette="husl")
    sb.pairplot(df, hue='area_type', diag_kind="kde", kind="scatter", palette="husl", plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
    # plt.show()
    return plt

# Pair wise plot
study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",")

# Different facet plots:
df_plots = [
    [['network_distance', 'efficiency', 'node_straightness', 'eccentricity', 'avg_shortest_path_duration', 'area_type'], 'random'],
    [['network_distance', 'node_d_km', 'edge_d_km', 'intersection_d_km', 'street_d_km', 'area_type'], 'densities'],
    [['network_distance', 'population', 'trips', 'area', 'circuity_avg', 'area_type'], 'dimensionless'],
    [['network_distance', 'avg_degree_connectivity', 'avg_edge_density', 'degree_centrality', 'avg_neighbor_degree', 'area_type'], 'degrees'],
    [['network_distance', 'clustering*', 'node_betweenness*', 'edge_betweenness', 'street_d_km', 'area_type'], 'dicts'],
    [['network_distance', 'btw_home_trip_production', 'btw_empl_trip_generation', 'btw_acc_trip_generation', 'btw_acc_trip_production', 'area_type'], 'btw_acc']
    # [['network_distance', 'avg_degree_connectivity', 'intersection_d_km', 'avg_neighbor_degree', 'area_type'], 'btw_acc']
            ]

# for df_selection in df_plots:
#     # fig = facet_plot(df[df_selection[0]])
#     fig = grid_plot(df[df_selection[0]], df_selection[1])
#
#     # Save plot
#     save_plot(fig, 'C:/Users/Ion/TFM/data/plots/facetPlot', df_selection[1])


# -----------------------------------------------------------------------------
# CORRELATION MATRIX
# -----------------------------------------------------------------------------
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.use('agg') #only save
# plt.use('TkAgg') #see plot
import os

def correlation_matrix(df):
    plt.figure(figsize=(28, 16))
    plt.rcParams.update({'font.size': 10})

    corrMatrix = df.corr()
    sb.heatmap(corrMatrix, annot=True)
    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.show()

    save_plot(plt, 'C:/Users/Ion/TFM/data/plots/attribute_plots/correlation_matrix', 'correlation_matrix')


# study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
# df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')
# df.drop(['area_type'], axis=1, inplace=True)
#
# correlation_matrix(df)



# -----------------------------------------------------------------------------
# PLOT OF INTEGER ATTRIBUTES
# -----------------------------------------------------------------------------

# study_area_dir = r'C:/Users/Ion/TFM/data/study_areas'
# # attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
# attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area', dtype=object)
# int_attr = ['network_distance',
#             'population',
#             'trips',
#             'n_intersection',
#             'node_d_km',
#             'intersection_d_km',
#             'edge_d_km',
#             'circuity_avg',
#             'edge_d_km']
# #
# rural = ['neuchatel', 'freiburg', 'plateau']
# mountain = ['chur', 'sion', 'linthal', 'frutigen', 'zermatt', 'locarno']
# urban = ['luzern', 'bern', 'zurich_kreis', 'lausanne', 'lugano', 'stgallen']
# all = rural + urban + mountain
#
# rural_df = attr_df.loc[rural]
# urban_df = attr_df.loc[urban]
# all_df = attr_df.loc[all]
#
# for df in [rural_df, urban_df, all_df]:
#     for attr in int_attr:
#         list_values = [int(d) for d in attr_df[attr].tolist()]
#         labels = df.index
#
# rural = ['sion', 'linthal', 'frutigen', 'neuchatel', 'zermatt', 'locarno', 'plateau']
# list_values = [float(d) for d in rural_df['edge_d_km'].tolist()]
#
# ax = plt.gca()
# plt.plot(rural, list_values)
#



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




