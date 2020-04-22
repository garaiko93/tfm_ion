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
import ntpath
import datetime
from scipy.optimize import curve_fit
from scipy import stats
import copy


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
'''
attr = 'btw_acc_trip_production'
env = 'All'
data_setup(r'C:/Users/Ion/TFM/data/study_areas/', attr, attr_dict[attr], env)

for env in ['All', 'Rural', 'Urban', 'Mountain']:
    for attr in list(attr_dict):
        data_setup('C:/Users/Ion/TFM/data/study_areas/', attr, attr_dict[attr], env)
'''

# -----------------------------------------------------------------------------
# GRID PLOT OF ATTRIBUTES
# -----------------------------------------------------------------------------
# https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
# https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0

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


def grid_plot(df, fig_name):
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

    # grid = grid.add_legend(fontsize=14, ncol=4, bbox_to_anchor=(0.5, 0.01), title='')#title='Area type',
    grid = grid.add_legend(fontsize=14, title='Area type')#title='Area type',
    # plt.legend(fontsize=14, ncol=3)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
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

'''
# Pair wise plot
study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",")

# Different facet plots:
df_plots = [
    # [['network_distance', 'efficiency', 'node_straightness', 'eccentricity', 'avg_shortest_path_duration', 'area_type'], 'random'],
    # [['network_distance', 'node_d_km', 'edge_d_km', 'intersection_d_km', 'street_d_km', 'area_type'], 'densities'],
    # [['network_distance', 'population', 'trips', 'area', 'circuity_avg', 'area_type'], 'dimensionless'],
    # [['network_distance', 'avg_degree_connectivity', 'avg_edge_density', 'degree_centrality', 'avg_neighbor_degree', 'area_type'], 'degrees'],
    # [['network_distance', 'clustering*', 'node_betweenness*', 'edge_betweenness', 'street_d_km', 'area_type'], 'dicts'],
    # [['network_distance', 'btw_home_trip_production', 'btw_empl_trip_generation', 'btw_acc_trip_generation', 'btw_acc_trip_production', 'area_type'], 'btw_acc']
    [['network_distance', 'node_d_km', 'edge_d_km', 'intersection_d_km', 'street_d_km', 'area_type'], 'try']
    # [['network_distance', 'avg_degree_connectivity', 'intersection_d_km', 'avg_neighbor_degree', 'area_type'], 'btw_acc']
            ]

for df_selection in df_plots:
    # fig = facet_plot(df[df_selection[0]])
    fig = grid_plot(df[df_selection[0]], df_selection[1])
#
#     # Save plot
    save_plot(fig, 'C:/Users/Ion/TFM/data/plots/attribute_plots/facetPlot', df_selection[1])
'''

# -----------------------------------------------------------------------------
# CORRELATION MATRIX
# -----------------------------------------------------------------------------
def correlation_matrix(df):
    plt.figure(figsize=(28, 16))
    plt.rcParams.update({'font.size': 10})

    corrMatrix = df.corr()
    sb.heatmap(corrMatrix, annot=True)
    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.show()

    save_plot(plt, 'C:/Users/Ion/TFM/data/plots/attribute_plots/correlation_matrix', 'correlation_matrix')

'''
study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')
df.drop(['area_type'], axis=1, inplace=True)

correlation_matrix(df)
'''



# -----------------------------------------------------------------------------
# PLOT SIM_RESULTS
# -----------------------------------------------------------------------------
def tend_curve(x, y, column=None):
    def exponential(x, a, k, b):
        return a * np.exp(-k * x) + b

    def inv_exponential(y, a, k, b):
        return (1 / k) * np.log(a / (y - b))

    def quadratic(x, a, k, b):
        return a * x ** 2 + k * x + b

    def inv_quadratic(y, a, k, b):
        # calculate the discriminant
        c = b - y
        d = (k ** 2) - (4 * a * c)

        # find two solutions
        sol1 = (-k - np.sqrt(d)) / (2 * a)
        sol2 = (-k + np.sqrt(d)) / (2 * a)

        if sol1 < 0:
            return sol2
        else:
            return sol1

    def error_f(x, y, model_func):
        try:
            opt, pcov = curve_fit(model_func, x, y, p0=(1., 1.e-5, 1.), maxfev=10000)
            a, k, b = opt
            error = list(abs(model_func(x, a, k, b) - np.array(y)))
            avg_error = sum(error) / len(error)
            # print(avg_error)
        except:
            avg_error = 10000
        return avg_error

    # Select which (exponential/quadratic) curves have less error on the fitting
    exp_error = error_f(x, y, exponential)
    quad_error = error_f(x, y, quadratic)

    if exp_error > quad_error:
        model_func = quadratic
        inv_func = inv_quadratic
        check_q = True
        print('quadratic better')
    else:
        model_func = exponential
        inv_func = inv_exponential
        check_q = False
        print('exponential better')

    # Fit only with exponential curve
    # model_func = exponential
    # inv_func = inv_exponential

    opt, pcov = curve_fit(model_func, x, y, p0=(1., 1.e-5, 1.), maxfev=10000)
    a, k, b = opt

    # Get threshold value of the fitted curve for a threshold of 300 wt
    if column is not None:
        opt_fs = int(np.rint(inv_func(300, a, k, b)))
    else:
        opt_fs = None

    # define limits to plot
    if check_q and opt_fs is not None:
        # this way, only one side of the parabola is plotted, until the derivate is max/min
        dx = -k / (2 * a)
        if dx > min(x) and dx < max(x):
            if opt_fs < dx:
                x2 = np.linspace(min(x), dx, 1000)
            else:
                x2 = np.linspace(dx, max(x), 1000)
        else:
            x2 = np.linspace(min(x), max(x), 1000)
        y2 = model_func(x2, a, k, b)
    else:
        x2 = np.linspace(min(x), max(x), 1000)
        y2 = model_func(x2, a, k, b)
    if column is not None:
        print('for: ' + str(column) + ' fleet size of: ' + str(opt_fs))

    return x2, y2, opt_fs

    # fig, ax = plt.subplots()
    # ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.9f x} %+.3f$' % (a, k, b))
    # # ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.9f x^2 + %.3f x + %+.3f$' % (a, k, b))
    # ax.plot(x, y, 'bo', label='data with noise')
    # ax.legend(loc='best')
    # plt.title('Distribution of trips in study area')
    # plt.xlabel('trip length (in seconds)')
    # plt.ylabel('count (normalized)')
    # plt.show()


def df_update(study_area_path, av_share, wt, area):
    if os.path.isfile(str(study_area_path) + '/sim_threshold.csv'):
        df = pd.read_csv(str(study_area_path) + '/sim_threshold.csv', sep=",", index_col='area')
    else:
        df = pd.DataFrame(data=None)

    try:
        # Save actual simulation value in df
        df.at[area, av_share] = wt
    except:
        # Update attribute table with new added attributes or study areas
        if area not in df.index:
            s = pd.Series(name=area)
            df = df.append(s)
        # create empty row with areas name to add attributes
        if av_share not in df.columns:
            df.insert(loc=len(df.columns), column=av_share, value=['' for i in range(df.shape[0])])
        # Save actual simulation value in df
        df.at[area, av_share] = int(wt)

    # Finally save df back
    df.to_csv(str(study_area_path) + '/sim_threshold.csv', sep=",", index=True, index_label='area')


# area_path = r'C:\Users\Ion\TFM\data\study_areas/locarno'
def plot_fc(area_path):
    print(datetime.datetime.now(), 'Creating plot, fitting curve with output waiting time average values ...')
    area = ntpath.split(area_path)[1]
    study_area_path = ntpath.split(area_path)[0]
    data_path = ntpath.split(study_area_path)[0]

    df_name = 'avg_df'
    df = pd.read_csv(str(area_path) + '/simulations/' + str(df_name) + '.csv', sep=",", index_col='fleet_size')

    # style
    plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 18})
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('tab10')

    # multiple line plot
    num = 0
    for column in df:
        # fit curve out of simulation results
        y = df[column]
        nan_elems = y.isnull()
        y = y[~nan_elems]
        for ix, wt in y.iteritems():
            if wt > 800 or wt < 100:
                y = y.drop(labels=[ix])
        x = y.index

        # xx, yy, opt_fs = tend_curve(df, column)
        xx, yy, opt_fs = tend_curve(x, y, column)

        # Save opt fs value for 300s waiting time in dataframe
        # df_update(study_area_path, column, opt_fs, area)
        x = df.index
        y = df[column]
        plt.scatter(x, y, color=palette(num), label=column)
        plt.plot(xx, yy, color=palette(num), linestyle='dashed')
        num += 1
        if column == '1.0':
            nan_elems = y.isnull()
            y = y[~nan_elems]
            for ix, wt in y.iteritems():
                if wt > 800 or wt < 100:
                    y = y.drop(labels=[ix])
            x = y.index
            thres_coord = max(x)

    # Add legend
    plt.legend(loc=0, ncol=1, prop={'size': 18}, frameon=True, title='AV share value:')

    # Define axis
    left, right = plt.xlim()
    # bottom, top = plt.ylim()
    ax = plt.axes()
    ax.set_ylim(125, 600)
    ax.set_xlim(left, right)

    # Add hline at wt 300 s
    plt.hlines(y=300, xmin=left, xmax=right, colors='r', linestyles='dashdot', zorder=100, label='Threshold')
    plt.text(thres_coord, 310, 'Threshold ', ha='right', va='center', fontsize=14, color='r')
    # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(formatter))

    # Add titles
    # plt.title('AV simulation results for: ' + str(area) + ' (10pct swiss census)', fontsize=16, fontweight=0,
    #           color='orange')
    plt.xlabel("AVs Fleet Size", fontsize=16)
    plt.ylabel("Waiting Time (s)", fontsize=16)

    # plt.savefig(str(area_path) + '/simulations/' + str(df_name) + '.png')

    plot_path = str(data_path) + '/plots/sim_plots'
    plt.savefig(str(plot_path) + '/' + str(area) + '.png')


# Parsing command line arguments
# parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
# parser.add_argument('--area-path', dest="area_path", help='path to simulation_output folder')
# args = parser.parse_args()
#
# area_path = args.area_path.split('\r')[0]

# plot_fc(area_path)

'''
study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'

study_area_list = list(os.walk(study_area_dir))[0][1]
study_area_list = ['bern']

for area in study_area_list:
    if area == 'test_area':
        continue
    area_path = str(study_area_dir) + '/' + str(area)
    plot_fc(area_path)
'''

# -----------------------------------------------------------------------------
# PLOT OPT_FS
# -----------------------------------------------------------------------------
def plot_opt_fs(df, path, ylabel='AVs Fleet Size for WaitingTime = 300 sec', cmap='gist_rainbow'):
    print(datetime.datetime.now(), 'Creating plot ...')
    # style
    plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 14})
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    NUM_COLORS = len(df)
    palette = plt.get_cmap(cmap)
    # palette = plt.get_cmap('tab20')
    # palette = plt.get_cmap('terrain')
    # palette = plt.get_cmap('gist_rainbow')
    # palette = plt.get_cmap('tab10')

    # multiple line plot
    num = 0
    for ix in df.index:
        # fit curve out of simulation results
        x = [20, 40, 60, 80, 100]
        y = list(df.loc[ix])
        xx, yy, opt_fs = tend_curve(x, y)

        # plt.scatter(x, y, color=palette(num), label=ix)
        # plt.plot(xx, yy, color=palette(num), linestyle='dashed')
        plt.scatter(x, y, color=palette(1. * num / NUM_COLORS), label=ix)
        plt.plot(xx, yy, color=palette(1. * num / NUM_COLORS), linestyle='dashed')
        num += 1

    # Add legend
    plt.legend(loc='best', ncol=1, prop={'size': 12}, frameon=True, title='Study Area')

    # Define axis
    ax = plt.axes()
    ax.tick_params(labelright=True)
    # left, right = plt.xlim()
    # bottom, top = plt.ylim()
    # ax.set_ylim(0, 6000)
    # ax.set_xlim(left, right)

    # Add titles
    # plt.title('AV simulation results for: ' + str(area) + ' (10pct swiss census)', fontsize=16, fontweight=0,
    #           color='orange')
    plt.xlabel("Car / PT users who switch to AV (%)", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    # save_plot(plt, path, 'opt_fs_plot_' + str(cmap))
    save_plot(plt, path, 'opt_fs_plot')
    # save_plot(plt, str(data_path) + '/plots/sim_plots/opt_fs', 'opt_fs_plot')
    # plt.savefig(str(data_path) + '/plots/sim_plots/opt_fs_plot/opt_fs_plot.png')


study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
data_path = ntpath.split(study_area_dir)[0]
df = pd.read_csv(str(study_area_dir) + '/' + 'sim_threshold.csv', sep=",", index_col='area')
df_N = pd.read_csv(str(study_area_dir) + '/' + 'sim_threshold_N.csv', sep=",", index_col='area')

df_N = df_N.sort_values(by=['1.0'], ascending=False)
df = df.sort_values(by=['1.0'], ascending=False)

# dfs = [df, df[df['1.0'] > 3000], df[df['1.0'] <= 3000]]

dfs = [[df, str(data_path) + '/plots/sim_plots/opt_fs/fs', 'AVs Fleet Size for WaitingTime = 300 sec'],
       [df[df['1.0'] > 3000], str(data_path) + '/plots/sim_plots/opt_fs/fs', 'AVs Fleet Size for WaitingTime = 300 sec'],
       [df[df['1.0'] <= 3000], str(data_path) + '/plots/sim_plots/opt_fs/fs', 'AVs Fleet Size for WaitingTime = 300 sec'],
       [df_N, str(data_path) + '/plots/sim_plots/opt_fs/normalized', 'Normalized AVs Fleet size']
       ]

# dfs = [[df_N, str(data_path) + '/plots/sim_plots/opt_fs/normalized', 'Normalized AVs Fleet size']]
cmaps = ['gist_rainbow', 'tab20', 'terrain', 'tab10', 'tab20b', 'rainbow', 'gnuplot2', 'gnuplot']
for df, path, ylabel in dfs:
    plot_opt_fs(df, path, ylabel, 'tab20b')


# -----------------------------------------------------------------------------
# PLOT OPT_FS VS ATTRIBUTES
# -----------------------------------------------------------------------------
def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)

#
# study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
# data_path = ntpath.split(study_area_dir)[0]
# sim_df = pd.read_csv(str(study_area_dir) + '/' + 'sim_threshold.csv', sep=",", index_col='area')
# sim_df = sim_df.sort_values(by=['1.0'], ascending=False)
# attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')
#
# sim_df100 = sim_df['1.0']
#
# # merge both df
# full_df = pd.concat([sim_df, attr_df], axis=1)
# full_df = full_df.sort_values(by=['network_distance'], ascending=False)
#
# # Normalization of fs
# sim_df_N = copy.deepcopy(sim_df)
# for area in sim_df_N.index:
#     carpt_users = int(attr_df.loc[area, 'CarPt_users'])
#     for av_share in sim_df_N.columns:
#         fs = int(sim_df_N.loc[area, av_share])
#         norm_value = fs / carpt_users
#
#         sim_df_N.at[area, av_share] = norm_value
# sim_df_N.to_csv(str(study_area_dir) + '/sim_threshold_N.csv', sep=",", index=True, index_label='area')





# Create new df with columns: ['area', 'area_type', 'attr_name', 'attr_value', 'fs']






# Plot map
# grid = sb.FacetGrid(df, hue="area_type", col="attr", height=4)
# grid = grid.map(qqplot, "attr_value", "fs")
# grid = grid.map(corr, corre=True)
# grid = grid.add_legend(fontsize=14, title='Area type')




# mng = plt.get_current_fig_manager()
# mng.frame.Maximize(True)
# plt.show()


# if sim_ready:
# area_path = r'C:\Users\Ion\TFM\data\study_areas/zurich_kreis'
# df_name='avg_df'
# df = pd.read_csv(str(area_path) + '/simulations/' + str(df_name) + '.csv', sep=",", index_col='fleet_size')
# df = df.fillna(0).groupby(level="fleet_size").sum()
# df.replace(0, np.nan, inplace=True)
# print(df)
# xlabel = 'Fleet size', ylabel = 'Waiting time (s)'

# xs = np.linspace(min(df.index), max(df.index), len(df.index))
# horiz_line_data = np.array([300 for i in range(max(df.index))])
#
# df.plot(ylim=[160,500], title='Simulation results for: ' + str(area))
# # plt.plot(xs, horiz_line_data, 'r--')
# plt.show()


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




