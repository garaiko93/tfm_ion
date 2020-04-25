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
from numpy import ones,vstack
from numpy.linalg import lstsq


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

'''
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
    grid = sb.PairGrid(data=df, hue='area_type', palette="husl", diag_sharey=False)
    # grid = sb.PairGrid(data=df, hue='area_type', palette="husl", height=4, aspect=1)
    # grid = sb.PairGrid(data=df)

    # Map the plots to the locations
    # grid = grid.map_upper(plt.scatter)
    # grid = grid.map_upper(plt.plot, color='none')
    grid = grid.map_upper(plt.plot, marker = "o", ls="")
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


# Pair wise plot
# study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
# df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",")
#
# # Different facet plots:
# df_plots = [
#     [['network_distance', 'efficiency', 'node_straightness', 'eccentricity', 'avg_shortest_path_duration', 'area_type'], 'random'],
#     [['network_distance', 'node_d_km', 'edge_d_km', 'intersection_d_km', 'street_d_km', 'area_type'], 'densities'],
#     [['network_distance', 'population', 'trips', 'area', 'circuity_avg', 'area_type'], 'dimensionless'],
#     [['network_distance', 'avg_degree_connectivity', 'avg_edge_density', 'degree_centrality', 'avg_neighbor_degree', 'area_type'], 'degrees'],
#     [['network_distance', 'clustering*', 'node_betweenness*', 'edge_betweenness', 'street_d_km', 'area_type'], 'dicts'],
#     [['network_distance', 'btw_home_trip_production', 'btw_empl_trip_generation', 'btw_acc_trip_generation', 'btw_acc_trip_production', 'area_type'], 'btw_acc']
#     # [['network_distance', 'node_d_km', 'edge_d_km', 'intersection_d_km', 'street_d_km', 'area_type'], 'try']
#     # [['network_distance', 'avg_degree_connectivity', 'intersection_d_km', 'avg_neighbor_degree', 'area_type'], 'btw_acc']
#             ]
#
# for df_selection in df_plots:
#     # fig = facet_plot(df[df_selection[0]])
#     fig = grid_plot(df[df_selection[0]], df_selection[1])
# #
# #     # Save plot
#     save_plot(fig, 'C:/Users/Ion/TFM/data/plots/attribute_plots/facetPlot', df_selection[1])


# -----------------------------------------------------------------------------
# CORRELATION MATRIX
# -----------------------------------------------------------------------------
def correlation_matrix(study_area_dir):
    df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')
    df.drop(['area_type'], axis=1, inplace=True)

    plt.figure(figsize=(28, 16))
    plt.rcParams.update({'font.size': 10})

    corrMatrix = df.corr()
    sb.heatmap(corrMatrix, annot=True)
    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.show()

    save_plot(plt, 'C:/Users/Ion/TFM/data/plots/attribute_plots/correlation_matrix', 'correlation_matrix')



# -----------------------------------------------------------------------------
# PLOT SIM_RESULTS
# -----------------------------------------------------------------------------
def tend_curve(x, y, func='exp', column=None):
    def exponential(x, a, k, b):
        return a * np.exp(-k * x) + b

    def inv_exponential(y, a, k, b):
        return (1 / k) * np.log(a / (y - b))

    # ---------------
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

    # ---------------
    def power(x, a, k, b):
        return a ** ((b * x) + k)

    def inv_power(y, a, k, b):
        return (1/b) * ((np.log(y) / np.log(a)) - k)

    # ---------------
    def two_points(x, a, k, b):
        return b*x + a
    def inv_two_points(y, a, k, b):
        return (y - a)/b
    # ---------------

    def plot_curve(x, model_func, a, k, b, opt_fs=None, check_q=False):
        # define limits to plot
        if check_q:
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
        return x2, y2

    def error_f(x, y, model_func):
        try:
            opt, pcov = curve_fit(model_func, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
            a, k, b = opt
            error = list(abs(model_func(x, a, k, b) - np.array(y)))
            avg_error = sum(error) / len(error)
            # print(avg_error)
        except:
            avg_error = 10000
        return avg_error

    # Select which (exponential/quadratic) curves have less error on the fitting
    # exp_error = error_f(x, y, exponential)
    # quad_error = error_f(x, y, quadratic)

    # Proceed depending on selected function
    if func == 'exp':
        opt, pcov = curve_fit(exponential, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
        a, k, b = opt

        x2, y2 = plot_curve(x, exponential, a, k, b, False)
        opt_fs = int(np.rint(inv_exponential(300, a, k, b)))

    elif func == 'qudratic':
        opt, pcov = curve_fit(exponential, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
        a, k, b = opt

        opt_fs = int(np.rint(inv_quadratic(300, a, k, b)))
        x2, y2 = plot_curve(x, quadratic, a, k, b, opt_fs, True)

    elif func == 'power':
        opt, pcov = curve_fit(power, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
        a, k, b = opt

        x2, y2 = plot_curve(x, power, a, k, b, False)
        opt_fs = int(np.rint(inv_power(300, a, k, b)))

    elif func == 'two_points':
        i = 0
        while y.iloc[i] > 300:
            i += 1
        upper = (x[i - 1], y.iloc[i - 1])
        lower = (x[i], y.iloc[i])

        points = [upper, lower]

        x_coords, y_coords = zip(*points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        b, a = lstsq(A, y_coords)[0]
        k = 0
        # print("Line Solution is y = {b}x + {a}".format(b=b, a=a))

        x2, y2 = plot_curve(x, two_points, a, k, b, check_q=False)
        opt_fs = int(np.rint(inv_two_points(300, a, k, b)))
    else:
        # Select which (exponential/quadratic) curves have less error on the fitting
        exp_error = error_f(x, y, exponential)
        quad_error = error_f(x, y, quadratic)
        power_error = error_f(x, y, power)
        if exp_error > quad_error and power_error > quad_error:
            opt, pcov = curve_fit(quadratic, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
            a, k, b = opt

            opt_fs = int(np.rint(inv_quadratic(300, a, k, b)))
            x2, y2 = plot_curve(x, quadratic, a, k, b, opt_fs, True)
            print('quadratic better')
        elif quad_error > exp_error and power_error > exp_error:
            opt, pcov = curve_fit(exponential, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
            a, k, b = opt

            opt_fs = int(np.rint(inv_exponential(300, a, k, b)))
            x2, y2 = plot_curve(x, exponential, a, k, b, opt_fs, False)
            print('exponential better')
        elif quad_error > power_error and exp_error > power_error:
            opt, pcov = curve_fit(power, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
            a, k, b = opt

            x2, y2 = plot_curve(x, power, a, k, b, False)
            opt_fs = int(np.rint(inv_power(300, a, k, b)))
            print('power better')
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


def df_update(threshold_path, av_share, wt, area):
    def save_value(df, area, av_share, wt):
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
        return df

    if os.path.isfile(threshold_path):
        df = pd.read_csv(threshold_path, sep=",", index_col='area')
    else:
        df = pd.DataFrame(data=None)

    if len(wt) == 1:
        df = save_value(df, area, av_share, wt)
    else:
        for i in range(len(wt)):
            df = save_value(df, area, av_share[i], wt[i])

    # Finally save df back
    df.to_csv(threshold_path, sep=",", index=True, index_label='area')


def plot_fc(area_path, plot_path, threshold_path, fit_func='exp'):
    print(datetime.datetime.now(), 'Creating plot, fitting curve with output waiting time average values ...')
    area = ntpath.split(area_path)[1]
    study_area_path = ntpath.split(area_path)[0]
    # data_path = ntpath.split(study_area_path)[0]

    df_name = 'avg_df'
    df = pd.read_csv(str(area_path) + '/simulations/' + str(df_name) + '.csv', sep=",", index_col='fleet_size')
    df = df[['0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '1.0']]
    df = df.sort_index(ascending=True)

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
            if wt > 600 or wt < 100:
                y = y.drop(labels=[ix])
        x = y.index

        # xx, yy, opt_fs = tend_curve(df, column)
        xx, yy, opt_fs = tend_curve(x, y, fit_func, column)

        # Save opt fs value for 300s waiting time in dataframe
        df_update(threshold_path, column, [opt_fs], area)

        plt.scatter(x, y, color=palette(num), label=column)
        plt.plot(xx, yy, color=palette(num), linestyle='dashed')
        num += 1
        if column == '1.0':
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

    # plot_path = str(data_path) + '/plots/sim_plots/try'
    # plt.savefig(str(plot_path) + '/' + str(area) + '.png')
    plt.savefig(plot_path)

def plot_sim_results(study_area_dir, plot_path, fit_func):
    study_area_list = list(os.walk(study_area_dir))[0][1]
    # study_area_list = ['bern']

    for area in study_area_list:
        if area == 'test_area':
            continue
        area_path = str(study_area_dir) + '/' + str(area)
        plot_fc(area_path,
                str(plot_path) + '/' + str(area) + '.png',
                str(plot_path) + '/' + 'sim_opt_fs.csv',
                fit_func=fit_func)  # fit_func = ['exp', 'quadratic', 'two_points', 'power' (a^(bx+c)), 'best' (better[exp,quadratic,power]) ]

    # Normalize output opt fs values and create a new csv
    sim_df_N = pd.read_csv(str(plot_path) + '/' + 'sim_opt_fs.csv', sep=",", index_col='area')
    attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')
    for area in sim_df_N.index:
        carpt_users = int(attr_df.loc[area, 'CarPt_users'])
        for av_share in sim_df_N.columns:
            fs = int(sim_df_N.loc[area, av_share])
            norm_value = fs / carpt_users

            sim_df_N.at[area, av_share] = norm_value

    sim_df_N = sim_df_N.sort_values(by=['1.0'], ascending=False)
    sim_df_N.to_csv(str(plot_path) + '/sim_opt_fs_norm.csv', sep=",", index=True, index_label='area')

# Parsing command line arguments
# parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
# parser.add_argument('--area-path', dest="area_path", help='path to simulation_output folder')
# args = parser.parse_args()
#
# area_path = args.area_path.split('\r')[0]

# plot_fc(area_path)



# -----------------------------------------------------------------------------
# PLOT OPT_FS
# -----------------------------------------------------------------------------
def plot_opt_fs(df, fig_name, ylabel, norm, study_area_dir, plot_path, regression, fit, cmap='gist_rainbow'):
    print(datetime.datetime.now(), 'Creating plot ...')
    # style
    plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 14})
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    NUM_COLORS = len(df)
    palette = plt.get_cmap(cmap)

    # multiple line plot
    num = 0
    for area in df.index:
        # fit curve out of simulation results
        x = [5, 10, 20, 40, 60, 80, 100]
        y = list(df.loc[area])

        if regression == 'linear':
            # fit linear regression and export a, b
            z = np.polyfit(x, y, 1)
            a, b = z

            # df_update(str(study_area_dir) + '/attr_table_AVG_T.csv', ['a', 'b'], [a, b], area)
            df_update(str(plot_path) + '/wt_fs/' + str(fit) + '/linear_regression.csv', ['a', 'b'], [a, b], area)
            # df_update(str(plot_path) + '/wt_fs/' + str(fit) + '/linear_regression_' + str(fit) + 'fit.csv', ['a', 'b'], [a, b], area)

            p = np.poly1d(z)
            plt.plot(x, p(x), color=palette(1. * num / NUM_COLORS), linestyle='dashed')

            plt.scatter(x, y, color=palette(1. * num / NUM_COLORS), label="%s: %sx + %s" % (area, "{:.5f}".format(a), "{:.5f}".format(b)))
        else:
            # fit exp curve
            xx, yy, opt_fs = tend_curve(x, y, func=regression)
            plt.plot(xx, yy, color=palette(1. * num / NUM_COLORS), linestyle='dashed')
            # plt.plot(xx, yy, color=palette(num), linestyle='dashed')
            plt.scatter(x, y, color=palette(1. * num / NUM_COLORS), label=area)

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
    plt.title('Fleet size / AV share value: ' + str(fit) + ' fit.', fontsize=16, fontweight=0,
              color='orange')
    plt.xlabel("Car / PT users who switch to AV (%)", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    save_plot(plt, str(plot_path) + '/fs_avshare/' + str(norm), str(fig_name) + '_' + str(fit) + 'fit')
    plt.savefig(str(plot_path) + '/wt_fs/' + str(fit) + '/' + str(fig_name) + '.png')

def plot_fs_avs_setup(study_area_dir, plot_path, regression, fit):
    # study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
    # data_path = ntpath.split(study_area_dir)[0]
    sim_path = str(plot_path) + '/wt_fs/' + str(fit)
    df = pd.read_csv(str(sim_path) + '/sim_opt_fs.csv', sep=",", index_col='area')
    df_N = pd.read_csv(str(sim_path) + '/sim_opt_fs_norm.csv', sep=",", index_col='area')

    df_N = df_N.sort_values(by=['1.0'], ascending=False)
    df = df.sort_values(by=['1.0'], ascending=False)

    # dfs = [[df, 'fs_avshare_all', 'AVs Fleet Size for WaitingTime = 300 sec'],
    #        [df[df['1.0'] > 3000], 'fs_avshare_1', 'AVs Fleet Size for WaitingTime = 300 sec'],
    #        [df[df['1.0'] <= 3000], 'fs_avshare_2', 'AVs Fleet Size for WaitingTime = 300 sec'],
    #        [df_N, 'fs_avshare_norm', 'Normalized AVs Fleet size']
    #        ]

    dfs = [[df, 'fs_avshare', 'AVs Fleet Size for WaitingTime = 300 sec', 'fs'],
           [df_N, 'fs_avshare_norm', 'Normalized AVs Fleet size', 'normalized']]

    cmaps = ['gist_rainbow', 'tab20', 'terrain', 'tab10', 'tab20b', 'rainbow', 'gnuplot2', 'gnuplot']
    for df, fig_name, ylabel, norm in dfs:
        plot_opt_fs(df, fig_name, ylabel, norm, study_area_dir, plot_path, regression, fit, 'tab20b')



# -----------------------------------------------------------------------------
# PLOT OPT_FS VS ATTRIBUTES
# -----------------------------------------------------------------------------
def qqplot(x, y, min, max,**kwargs):
    # _, xr = stats.probplot(x, fit=False)
    # _, yr = stats.probplot(y, fit=False)
    # plt.scatter(xr, yr, **kwargs)


    left = min.iloc[0] - (0.1 * (max.iloc[0] - min.iloc[0]))
    right = max.iloc[0] + (0.1 * (max.iloc[0] - min.iloc[0]))

    plt.xlim(left, right)
    plt.scatter(x, y, **kwargs)

    # If correlation is high enough, fit curve
    coef = np.corrcoef(x, y)[0][1]
    if coef > 0.5 or coef < -0.5:
        print(coef)
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        # plt.plot(x, p(x), linestyle='dashed')

        # xx, yy, opt_fs = tend_curve(xr, yr, func='exp')
        xx, yy, opt_fs = tend_curve(x, y, func='exp')
        plt.plot(xx, yy, linestyle='dashed', **kwargs)


def plot_fs_attr(full_df, attr_list, plot_path):
    # Create new df with columns: ['area', 'area_type', 'attr_name', 'attr_value', 'fs']
    data = []
    for attr in attr_list:
        if attr == 'area_type':
            continue
        for area in full_df.index:
            new_row = (area,
                       full_df.at[area, 'area_type'],
                       attr,
                       full_df.at[area, attr],
                       full_df.at[area, '1.0'],
                       min(full_df[attr]),
                       max(full_df[attr]))
            # if new_row[1] == 'urban':
            #     data.append(new_row)
            data.append(new_row)
    df = pd.DataFrame(data, columns=['area', 'area_type', 'Attribute', 'attr_value', 'fs_N', 'min', 'max'])

    # Plot map
    grid = sb.FacetGrid(df, hue="area_type", col_wrap=3, col="Attribute", height=4, sharex=False)
    grid = grid.map(qqplot, "attr_value", "fs_N", 'min', 'max')
    # grid = grid.map(corr, corre=True)
    grid = grid.add_legend(fontsize=14, title='Area type', frameon=True)

    save_plot(grid, plot_path, 'fsN_attr')

def plot_fs_attr_setup(study_area_dir, plot_path, attr_list):
    # study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
    # plot_path = 'C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/only_exp_l600'
    # data_path = ntpath.split(study_area_dir)[0]
    sim_df = pd.read_csv(str(study_area_dir) + '/' + 'sim_threshold.csv', sep=",", index_col='area')
    attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')
    sim_df_N = pd.read_csv(str(study_area_dir) + '/' + 'sim_threshold.csv', sep=",", index_col='area')

    # sim_df100 = sim_df['1.0']


    # merge both df
    # full_df = pd.concat([sim_df, attr_df], axis=1)
    full_df = pd.concat([sim_df_N, attr_df], axis=1, sort=False)
    full_df = full_df.sort_values(by=['network_distance'], ascending=False)

    # Call function to plot attributes
    if attr_list != None:
        plot_fs_attr(full_df, attr_list, plot_path)
    else:
        # Automate to plot all attributes
        count = 0
        attr_list = []
        for i in attr_df.columns:
            attr_list.append(i)
            if len(attr_list) == 9:
                plot_fs_attr(full_df, attr_list, plot_path)
                attr_list = []
            count += 1


# This creates plot for each area (wt vs fs) and opt fs csv (fs, and fs_N)
# plot_sim_results(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#                  plot_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/two_points',
#                  fit_func='exp')

# This creates for a defined fs or fs_N csv, the corresponding plot fs vs avshare value, with a regression (linear, exp)
plot_fs_avs_setup(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
                  plot_path='C:/Users/Ion/TFM/data/plots/sim_plots',
                  regression='linear', # ['linear' (saves a b as attributes), 'exp', 'power', 'quadratic', 'best']
                  fit='power')            # ['exp', 'power', 'best', 'two_points'] any already computed fit foldername of wt/fs



# plot predefined attributes:
# attr_list = ['node_d_km', 'edge_d_km', 'CarPt_users', 'trips',
#              'network_distance', 'circuity_avg', 'avg_degree', 'avg_neighbor_degree',
#              'degree_centrality', 'node_straightness', 'node_closeness_time*', 'efficiency',
#              'btw_home_trip_production', 'btw_empl_trip_generation', 'btw_acc_trip_generation', 'btw_acc_trip_production']
# attr_list = ['avg_degree', 'node_d_km']
# attr_list = ['node_closeness_time*']
#
#
# plot_fs_attr_setup(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#                    plot_path='C:/Users/Ion/TFM/data/plots/regression',
#                    attr_list=None)



# correlation_matrix('C:/Users/Ion/TFM/data/study_areas')



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




