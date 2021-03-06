import matplotlib as mpl
# mpl.use('agg') #only save
# mpl.use('TkAgg') #see plot
import pickle
import pandas as pd
# import tkinter
import seaborn as sb
from matplotlib import pyplot as plt
import os
import numpy as np
import ntpath
import datetime
from scipy.optimize import curve_fit
from scipy import stats
import copy
import matplotlib.ticker as ticker
from numpy import ones, vstack
from numpy.linalg import lstsq
# from brokenaxes import brokenaxes

# functions from other scripts
from vioboxPlot import violinboxplot

# Turn interactive plotting off
plt.ioff()

# # Program to show various ways to read and
# # write data in a file.
# file1 = open("path_val.txt", "w")
# # \n is placed to indicate EOL (End of Line)
# file1.write(os.environ['PATH'])
# file1.close()  # to change file access modes

def save_plot(fig, path, fig_name):
    fig_path = str(path) + '/' + str(fig_name) + '.png'
    if not os.path.isfile(fig_path):
        fig.savefig(fig_path)
    else:
        i = 1
        fig_path = str(path) + '/' + str(fig_name) + str(i) + '.png'
        while os.path.isfile(fig_path):
            i += 1
            fig_path = str(path) + '/' + str(fig_name) + str(i) + '.png'
        fig.savefig(fig_path)
    print('Plot saved: ' + str(fig_name))

def align_axis(ax1, ax2, step=1):
    """ Sets both axes to have the same number of gridlines
        ax1: left axis
        ax2: right axis
        step: defaults to 1 and is used in generating a range of values to check new boundary
              as in np.arange([start,] stop[, step])
    """
    ax1.set_aspect('auto')
    ax2.set_aspect('auto')

    grid_l = len(ax1.get_ygridlines())  # N of gridlines for left axis
    grid_r = len(ax2.get_ygridlines())  # N of gridlines for right axis
    grid_m = max(grid_l, grid_r)  # Target N of gridlines

    #  Choose the axis with smaller N of gridlines
    if grid_l < grid_r:
        y_min, y_max = ax1.get_ybound()  # Get current boundaries
        parts = (y_max - y_min) / (grid_l - 1)  # Get current number of partitions
        left = True
    elif grid_l > grid_r:
        y_min, y_max = ax2.get_ybound()
        parts = (y_max - y_min) / (grid_r - 1)
        left = False
    else:
        return None

    # Calculate the new boundary for axis:
    yrange = np.arange(y_max + 1, y_max * 2 + 1, step)  # Make a range of potential y boundaries
    parts_new = (yrange - y_min) / parts  # Calculate how many partitions new boundary has
    y_new = yrange[np.isclose(parts_new, grid_m - 1)]  # Find the boundary matching target

    # Set new boundary
    print(y_new)
    if left:
        return ax1.set_ylim(top=round(y_new, 0), emit=True, auto=True)
    else:
        return ax2.set_ylim(top=round(y_new, 0), emit=True, auto=True)

# -----------------------------------------------------------------------------
# PLOT OF DICTIONARY SHAPE ATTRIBUTES
# -----------------------------------------------------------------------------
# DATA PREPARATION TO PLOT
def data_setup(study_areas, attr, plot_title, list_areas):
    cap_areas = {'bern': 'Bern',
                 'chur': 'Chur',
                 'freiburg': 'Freiburg',
                 'frutigen': 'Frutigen',
                 'lausanne': 'Lausanne',
                 'linthal': 'Linthal',
                 'locarno': 'Locarno',
                 'lugano': 'Lugano',
                 'luzern': 'Luzern',
                 'neuchatel': 'Neuchatel',
                 'plateau': 'Plateau',
                 'sion': 'Sion',
                 'stgallen': 'St. Gallen',
                 'zermatt': 'Zermatt',
                 'zurich_kreis': 'Zürich'}
    if list_areas == 'All':
        areas = ['zermatt', 'locarno', 'chur', 'sion', 'linthal', 'frutigen', 'freiburg', 'neuchatel', 'plateau', 'luzern', 'bern',
                 'zurich_kreis', 'lausanne', 'lugano', 'stgallen']
    elif list_areas == 'Rural':
        areas = ['freiburg', 'neuchatel', 'plateau', 'stgallen']
    elif list_areas == 'Urban':
        areas = ['luzern', 'bern', 'zurich_kreis', 'lausanne', 'lugano']
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
        x_axis.append(cap_areas[area])

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

    violinboxplot(data_to_plot, labels=x_axis, ax=ax, showModes=True, showCounts=True, outliers=outliers, logPercentile=logPercentile, xtitle=plot_title)

    save_plot(plt, 'C:/Users/Ion/TFM/data/plots/attribute_plots/violinboxplot/' + str(list_areas), attr)
    # plt.savefig('C:/Users/Ion/TFM/data/plots/attribute_plots/' + str(list_areas) + '/' + str(attr) + '.png')
    # plt.savefig(r'C:/Users/Ion/TFM/' + str(attr_name) + '.png')
    print('Plot saved: ' + str(list_areas) + ' ' + str(attr))

'''
attr_dict = {
    'avg_degree_connect': 'Average degree connectivity',
    'avg_neighbor_degree': 'Average neighbour degree',
    'clustering': 'Clustering coefficient', #is not useful
    'degree_centrality': 'Degree centrality',
    'eccentricity': 'Eccentricity',
    'edge_betweenness': 'Edge betweenness centrality',
    # 'edge_load_centrality': 'Edge Load Centrality',
    'node_betweenness': 'Node betweenness centrality',
    'node_closeness_length': 'Node closeness centrality by distance',
    'node_closeness_time': 'Node closeness centrality by time',
    # 'node_load_centrality': 'Node Load Centrality',
    'node_straightness': 'Node straightness centrality',
    'btw_home_trip_production': 'Betweenness accessibility: Links trip production potential',
    'btw_empl_trip_generation': 'Betweenness accessibility: Links trip generation potential',
    'btw_acc_trip_generation': 'Betweenness accessibility: Links accessibility to population',
    'btw_acc_trip_production': 'Betweenness accessibility: Links accessibility to employment',
}

# attr = 'btw_acc_trip_production'
# env = 'All'
# data_setup(r'C:/Users/Ion/TFM/data/study_areas/', attr, attr_dict[attr], env)

for env in ['All', 'Rural', 'Urban', 'Mountain']:
    for attr in list(attr_dict):
        data_setup('C:/Users/Ion/TFM/data/study_areas/', attr, attr_dict[attr], env)
# '''

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
    # plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 16})

    grid = sb.PairGrid(data=df, hue='area_type', palette="husl", diag_sharey=False)
    # grid = sb.PairGrid(data=df, hue='area_type', palette="husl", height=4, aspect=1)
    # grid = sb.PairGrid(data=df)

    # Map the plots to the locations
    # grid = grid.map_upper(plt.scatter)
    # grid = grid.map_upper(plt.plot, color='none')
    grid = grid.map_upper(plt.plot, marker="o", ls="")
    grid = grid.map_upper(corr, corre=True)
    grid = grid.map_lower(sb.kdeplot)
    grid = grid.map_diag(plt.hist, bins=5, edgecolor='k')
    # grid = grid.map_diag(sb.distplot, hist=False)

    # grid = grid.add_legend(fontsize=14, ncol=4, bbox_to_anchor=(0.5, 0.01), title='')#title='Area type',
    # grid = grid.add_legend(fontsize=14, title='Area type')#title='Area type',
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

def setup_gridplot(study_area_dir, plot_path, df_plots, df=None):
    # Pair wise plot
    # study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
    # df = pd.read_csv(str(study_area_dir) + '/attribute_table_AVG_T.csv', sep=",")
    if df is None:
        df = pd.read_csv(str(study_area_dir) + '/full_df_two_points_norm.csv', sep=",")

    for df_selection in df_plots:
        # fig = facet_plot(df[df_selection[0]])
        fig = grid_plot(df[df_selection[0]], df_selection[1])

        # Save plot
        save_plot(fig, plot_path, df_selection[1])


# -----------------------------------------------------------------------------
# CORRELATION MATRIX
# -----------------------------------------------------------------------------
def correlation_matrix(study_area_dir, sim_path, fit):
    # Load dfs: [a,b], [sim_results], [attrbiutes]
    df_lr = pd.read_csv(str(sim_path) + '/' + 'linear_regression.csv', sep=",", index_col='area')
    df_sim = pd.read_csv(str(sim_path) + '/' + 'sim_opt_fs_norm.csv', sep=",", index_col='area')
    df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')
    df.drop(['area_type'], axis=1, inplace=True)

    full_df = pd.concat([df, df_sim, df_lr], axis=1, sort=False)
    full_df = full_df.sort_values(by=['network_distance'], ascending=False)

    for method in ['pearson', 'kendall', 'spearman']:
    # for method in ['spearman']:
        for pos in ['abs', 'natural']:
        # for pos in ['natural']:
            # plt.figure(figsize=(34, 22))
            # plt.figure(figsize=(40, 18))
            plt.figure(figsize=(38, 24))
            plt.rcParams.update({'font.size': 16})

            if pos == 'abs':
                corrMatrix = full_df.corr(method=method).abs().round(decimals=2)
            else:
                corrMatrix = full_df.corr(method=method).round(decimals=2)

            mask = np.zeros(corrMatrix.shape, dtype=bool)
            mask[np.triu_indices(len(mask))] = True

            sb.heatmap(corrMatrix, annot=True, mask=mask)
            plt.gcf().subplots_adjust(bottom=0.15)
            # plt.show()
            plt.tight_layout()

            # save_plot(plt, 'C:/Users/Ion/TFM/data/plots/attribute_plots/correlation_matrix/' + str(fit), 'correlation_matrix_' + str(method) + '_' + str(pos))
            save_plot(plt, 'C:/Users/Ion/TFM/data/plots/final/correlation_matrix', 'correlation_matrix_' + str(method) + '_' + str(pos))


# -----------------------------------------------------------------------------
# PLOT SIM_RESULTS
# -----------------------------------------------------------------------------
def tend_curve(x, y, func='exp', wt_threshold=None, column=None):
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

    # Proceed depending on selected function
    if func == 'exp':
        opt, pcov = curve_fit(exponential, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
        a, k, b = opt

        x2, y2 = plot_curve(x, exponential, a, k, b, False)
        opt_fs = int(np.rint(inv_exponential(wt_threshold, a, k, b)))

    elif func == 'qudratic':
        opt, pcov = curve_fit(exponential, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
        a, k, b = opt

        opt_fs = int(np.rint(inv_quadratic(wt_threshold, a, k, b)))
        x2, y2 = plot_curve(x, quadratic, a, k, b, opt_fs, True)

    elif func == 'power':
        opt, pcov = curve_fit(power, x, y, maxfev=20000, bounds=([-np.inf, 0, -np.inf],
                                                                 [np.inf, np.inf, np.inf]))
        a, k, b = opt
        print(opt)
        x2, y2 = plot_curve(x, power, a, k, b, False)
        opt_fs = int(np.rint(inv_power(wt_threshold, a, k, b)))

    elif func == 'two_points':
        i = 0
        while y.iloc[i] > wt_threshold:
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
        opt_fs = int(np.rint(inv_two_points(wt_threshold, a, k, b)))
    else:
        # Select which (exponential/quadratic) curves have less error on the fitting
        exp_error = error_f(x, y, exponential)
        quad_error = error_f(x, y, quadratic)
        power_error = error_f(x, y, power)
        if exp_error > quad_error and power_error > quad_error:
            opt, pcov = curve_fit(quadratic, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
            a, k, b = opt

            opt_fs = int(np.rint(inv_quadratic(wt_threshold, a, k, b)))
            x2, y2 = plot_curve(x, quadratic, a, k, b, opt_fs, True)
            print('quadratic better')
        elif quad_error > exp_error and power_error > exp_error:
            opt, pcov = curve_fit(exponential, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
            a, k, b = opt

            opt_fs = int(np.rint(inv_exponential(wt_threshold, a, k, b)))
            x2, y2 = plot_curve(x, exponential, a, k, b, opt_fs, False)
            print('exponential better')
        # elif quad_error > power_error and exp_error > power_error:
        else:
            opt, pcov = curve_fit(power, x, y, p0=(1., 1.e-5, 1.), maxfev=20000)
            a, k, b = opt

            x2, y2 = plot_curve(x, power, a, k, b, False)
            opt_fs = int(np.rint(inv_power(wt_threshold, a, k, b)))
            print('power better')

    if column is not None:
        print('for: ' + str(column) + ' fleet size of: ' + str(opt_fs))

    return x2, y2, opt_fs


def df_update(opt_fs_path, av_share, wt, area):
    def save_value(df, area, av_share, wt):
        if df.at[area, av_share] is not None:
            return df
        elif df.at[area, 1] is not None:
            return df
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

    if os.path.isfile(opt_fs_path):
        try:
            df = pd.read_csv(opt_fs_path, sep=",", index_col='study_area')
        except:
            df = pd.read_csv(opt_fs_path, sep=",", index_col='area')
    else:
        df = pd.DataFrame(data=None)

    if len(wt) == 1:
        df = save_value(df, area, av_share, wt[0])
    else:
        for i in range(len(wt)):
            df = save_value(df, area, av_share[i], wt[i])

    # Finally save df back
    df.to_csv(opt_fs_path, sep=",", index=True, index_label='study_area')


def plot_fc(area_path, plot_path, opt_fs_path, area, fit_func='exp'):
    print(datetime.datetime.now(), 'Creating plot, fitting curve with output waiting time average values ...')
    area = ntpath.split(area_path)[1]
    study_area_path = ntpath.split(area_path)[0]
    # data_path = ntpath.split(study_area_path)[0]

    df_name = 'avg_df'
    df = pd.read_csv(str(area_path) + '/simulations/' + str(df_name) + '.csv', sep=",", index_col='fleet_size')
    df = df[['0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '1.0']]
    df = df.sort_index(ascending=True)

    # style
    # plt.figure(figsize=(17, 8))
    plt.figure(figsize=(5, 3.5))
    plt.rcParams.update({'font.size': 18})
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('tab10')

    # multiple line plot
    num = 0
    wt_threshold = 5
    for column in df:
        # fit curve out of simulation results
        y = df[column]
        nan_elems = y.isnull()
        y = y[~nan_elems]

        if any(400 < flag < 600 for (flag) in y):
            for ix, wt in y.iteritems():
                if wt > 600 or wt < 100:
                    y = y.drop(labels=[ix])
            x = y.index
        else:
            for ix, wt in y.iteritems():
                if wt > 700 or wt < 100:
                    y = y.drop(labels=[ix])
            x = y.index

        # This is for y axis to appear in minutes instead of seconds
        y = y/60

        # xx, yy, opt_fs = tend_curve(df, column)
        xx, yy, opt_fs = tend_curve(x, y, fit_func, wt_threshold, column)

        # Save opt fs value for 300s waiting time in dataframe
        # df_update(opt_fs_path, column, [opt_fs], area)

        plt.scatter(x, y, color=palette(num), label=column)
        plt.plot(xx, yy, color=palette(num), linestyle='dashed')
        num += 1
        if column == '1.0':
            thres_coord = max(x)

    # Define axis
    left, right = plt.xlim()
    # bottom, top = plt.ylim()
    ax = plt.axes()
    ax.set_ylim(wt_threshold * 0.4, wt_threshold * 2)
    ax.set_xlim(left, right)

    # Define y-axis ticks
    plt.yticks([3,5,7,9], ['3','5','7','9'])

    # Add hline at wt 300 s
    plt.hlines(y=wt_threshold, xmin=left, xmax=right, colors='r', linestyles='dashdot', zorder=100, label='Threshold')
    # plt.text(thres_coord, wt_threshold + (wt_threshold - wt_threshold*0.4)*(1/18), 'Threshold ', ha='right', va='center', fontsize=14, color='r')
    # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(formatter))

    # Add legend
    # plt.legend(ncol=1, frameon=True, title='AV share value:', loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(loc=0, ncol=1, prop={'size': 18}, frameon=True, title='AV share value:')

    # Add titles
    # plt.title('AV simulation results for: ' + str(area) + ' (10pct swiss census)', fontsize=16, fontweight=0,
    #           color='orange')
    plt.xlabel("Fleet Size")
    plt.ylabel("Waiting Time [min]")
    plt.tight_layout()

    # plt.savefig(str(area_path) + '/simulations/' + str(df_name) + '.png')

    # plot_path = str(data_path) + '/plots/sim_plots/try'
    # plt.savefig(str(plot_path) + '/' + str(area) + '.png')
    # plt.savefig(plot_path)
    save_plot(plt, plot_path, area)

def plot_sim_results(study_area_dir, plot_path, fit_func, area=None):
    if area:
        study_area_list = [area]
    else:
        study_area_list = list(os.walk(study_area_dir))[0][1]

    for area in study_area_list:
        if area in ['test_area', 'baden', 'interlaken', 'geneve', 'basel', 'winterthur']:
            continue
        area_path = str(study_area_dir) + '/' + str(area)
        plot_fc(area_path,
                # plot_path,
                'C:/Users/Ion/TFM/data/plots/final/no_legend',
                str(plot_path) + '/' + 'sim_opt_fs.csv',
                area,
                fit_func=fit_func)  # fit_func = ['exp', 'quadratic', 'two_points', 'power' (a^(bx+c)), 'best' (better[exp,quadratic,power]) ]

    # Normalize output opt fs values and create a new csv
    # sim_df_N = pd.read_csv(str(plot_path) + '/' + 'sim_opt_fs.csv', sep=",", index_col='area')
    # attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')
    # for area in sim_df_N.index:
    #     carpt_users = int(attr_df.loc[area, 'CarPt_users'])
    #     for av_share in sim_df_N.columns:
    #         fs = int(sim_df_N.loc[area, av_share])
    #         norm_value = fs / carpt_users
    #
    #         sim_df_N.at[area, av_share] = norm_value
    #
    # sim_df_N = sim_df_N.sort_values(by=['1.0'], ascending=False)
    # sim_df_N.to_csv(str(plot_path) + '/sim_opt_fs_norm.csv', sep=",", index=True, index_label='area')

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
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 16})
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    NUM_COLORS = len(df)
    palette = plt.get_cmap(cmap)
    # multiple line plot
    num = 0

    # Define way_type for each area:
    area_type_dict = {'baden': 'rural',
                      'bern': 'urban',
                      'bern_large': 'urban',
                      'basel': 'urban',
                      'chur': 'mountain',
                      'freiburg': 'rural',
                      'frutigen': 'mountain',
                      'geneve': 'urban',
                      'interlaken': 'mountain',
                      'lausanne': 'urban',
                      'lausanne_lake': 'urban',
                      'linthal': 'mountain',
                      'locarno': 'mountain',
                      'lugano': 'urban',
                      'luzern': 'urban',
                      'neuchatel': 'rural',
                      'plateau': 'rural',
                      'sion': 'mountain',
                      'stgallen': 'rural',
                      'test_area': 'urban',
                      'winterthur': 'rural',
                      'zermatt': 'mountain',
                      'zurich_kreis': 'urban',
                      'zurich_large': 'urban'}
    area_type_list = []
    for area in df.index:
        # fit curve out of simulation results
        x = [5, 10, 20, 40, 60, 80, 100]
        y = list(df.loc[area])

        area_type = area_type_dict[area]
        if area_type == 'rural':
            # color = (255, 51, 150)
            color = '#EB0079'
        elif area_type == 'urban':
            # color = (51, 170, 255)
            color = '#00A3F1'
        elif area_type == 'mountain':
            # color = (1, 218, 7)
            color = '#0DC300'
        else:
            color = None

        if regression == 'linear':
            # fit linear regression and export a, b
            z = np.polyfit(x, y, 1)
            a, b = z

            # df_update(str(study_area_dir) + '/attr_table_AVG_T.csv', ['a', 'b'], [a, b], area)
            # df_update(str(plot_path) + '/wt_fs/' + str(fit) + '/linear_regression.csv', ['a', 'b'], [a, b], area)
            # df_update(str(plot_path) + '/wt_fs/' + str(fit) + '/linear_regression_' + str(fit) + 'fit.csv', ['a', 'b'], [a, b], area)

            p = np.poly1d(z)
            if area_type not in area_type_list:
                plt.plot(x, p(x), color=color, linestyle='dashed')
                plt.scatter(x, y, color=color, label=area_type)
                area_type_list.append(area_type)
            else:
                plt.plot(x, p(x), color=color, linestyle='dashed')
                plt.scatter(x, y, color=color)


            # plt.plot(x, p(x), color=palette(1. * num / NUM_COLORS), linestyle='dashed')
            # plt.scatter(x, y, color=palette(1. * num / NUM_COLORS), label="%s: %sx + %s" % (area, "{:.5f}".format(a), "{:.5f}".format(b)))
        else:
            # fit exp curve
            xx, yy, opt_fs = tend_curve(x, y, func=regression)
            plt.plot(xx, yy, color=palette(1. * num / NUM_COLORS), linestyle='dashed')
            # plt.plot(xx, yy, color=palette(num), linestyle='dashed')
            plt.scatter(x, y, color=palette(1. * num / NUM_COLORS), label=area)

        num += 1

    # Add legend
    plt.legend(loc='best', ncol=1, prop={'size': 15}, frameon=True, title='Area type')

    # Define axis
    ax = plt.axes()
    ax.tick_params(labelright=True)
    # left, right = plt.xlim()
    # bottom, top = plt.ylim()
    # ax.set_ylim(0, 6000)
    # ax.set_xlim(left, right)

    # Add titles
    # plt.title('Fleet size / AV share value: ' + str(fit) + ' fit.', fontsize=16, fontweight=0, color='orange')
    plt.xlabel("Car / PT users who switch to AV (%)", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    save_plot(plt, str(plot_path) + '/fs_avshare/' + str(norm), str(fig_name) + '_' + str(fit) + 'fit')
    save_plot(plt, str(plot_path) + '/wt_fs/' + str(fit), str(fig_name))
    # save_plot(plt, str(plot_path), str(fig_name) + '_' + str(fit) + 'fit')

    # plt.savefig(str(plot_path) + '/fs_avshare/' + str(norm) + '/' + str(fig_name) + '_' + str(fit) + 'fit.png')
    # plt.savefig(str(plot_path) + '/wt_fs/' + str(fit) + '/' + str(fig_name) + '.png')


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
def qqplot(x, y, min, max, **kwargs):
    # _, xr = stats.probplot(x, fit=False)
    # _, yr = stats.probplot(y, fit=False)
    # plt.scatter(xr, yr, **kwargs)


    left = min.iloc[0] - (0.1 * (max.iloc[0] - min.iloc[0]))
    right = max.iloc[0] + (0.1 * (max.iloc[0] - min.iloc[0]))

    plt.xlim(left, right)
    plt.scatter(x, y, **kwargs)

    # If correlation is high enough, fit curve
    coef = np.corrcoef(x, y)[0][1]
    if abs(coef) > 0.5:
        print(coef)
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        # plt.plot(x, p(x), linestyle='dashed')

        # xx, yy, opt_fs = tend_curve(xr, yr, func='exp')
        xx, yy, opt_fs = tend_curve(x, y, func='exp')
        plt.plot(xx, yy, linestyle='dashed', **kwargs)


def qqplot2(x, y, left, right, down, up, param, area_type=None, **kwargs):
    # print(x, y, left, right, down, up, param, area_type)
    if min(left.iloc[0], down.iloc[0]) < 0:
        low = min(left.iloc[0], down.iloc[0]) * 1.2
    else:
        low = min(left.iloc[0], down.iloc[0]) * 0.8
    high = max(right.iloc[0], up.iloc[0]) * 1.2

    plt.xlim(low, high)
    plt.ylim(low, high)

    plt.scatter(x, y, **kwargs)
    if param.iloc[0] == 'fleet_size':
        plt.yscale('log')
        plt.xscale('log')

    xx = np.linspace(low, high, 100)
    plt.plot(xx, xx, linestyle='dashed', c='r', linewidth=0.6)
    plt.plot(xx, xx*1.2, linestyle='dashed', c='g', linewidth=0.8)
    plt.plot(xx, xx*0.8, linestyle='dashed', c='g', linewidth=0.8, label='20% error')


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
    df = pd.DataFrame(data, columns=['study_area', 'area_type', 'Attribute', 'attr_value', 'fs_N', 'min', 'max'])

    # Plot map
    grid = sb.FacetGrid(df, hue="area_type", col_wrap=3, col="Attribute", height=4, sharex=False)
    grid = grid.map(qqplot, "attr_value", "fs_N", 'min', 'max')
    # grid = grid.map(corr, corre=True)
    grid = grid.add_legend(fontsize=14, title='Area type', frameon=True)

    save_plot(grid, plot_path, 'fsN_attr')

def plot_fs_attr_setup(study_area_dir, sim_path, plot_path, attr_list):
    # study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
    # plot_path = 'C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/only_exp_l600'
    # data_path = ntpath.split(study_area_dir)[0]
    # sim_df = pd.read_csv(str(sim_path) + '/' + 'sim_opt_fs.csv', sep=",", index_col='area')
    sim_df_N = pd.read_csv(str(sim_path) + '/' + 'sim_opt_fs_norm.csv', sep=",", index_col='area')
    attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')

    # sim_df100 = sim_df['1.0']

    # merge both df
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


def plot_pred_real(full_df, pred_list, plot_path):
    # Create new df with columns: ['area', 'area_type', 'attr_name', 'attr_value', 'fs']
    data = []
    for real, pred, parameter in pred_list:
        for area in full_df.index:
            if parameter == 'fleet_size':
                for avshare in ['0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '1.0']:
                    new_row = (area,
                               full_df.at[area, 'area_type'],
                               full_df.at[area, 'real_' + avshare],
                               full_df.at[area, 'pred_' + avshare],
                               parameter,
                               min(full_df['real_' + avshare]),
                               max(full_df['real_' + avshare]),
                               min(full_df['pred_' + avshare]),
                               max(full_df['pred_' + avshare]))
                    data.append(new_row)
            else:
                new_row = (area,
                           full_df.at[area, 'area_type'],
                           full_df.at[area, real],
                           full_df.at[area, pred],
                           parameter,
                           min(full_df[real]),
                           max(full_df[real]),
                           min(full_df[pred]),
                           max(full_df[pred]))
                data.append(new_row)
            # print(new_row)
    df = pd.DataFrame(data, columns=['study_area', 'area_type', 'real', 'pred', 'parameter', 'left', 'right', 'down', 'up'])
    # print(df.columns)
    # Plot map
    grid = sb.FacetGrid(df, hue="area_type", col_wrap=2, col="parameter", height=4, sharex=False, sharey=False)
    # grid = grid.map(qqplot2, 'pred', 'real', 'left', 'right', 'down', 'up')
    grid = grid.map(qqplot2, 'area_type', 'pred', 'real', 'left', 'right', 'down', 'up', 'parameter')
    # grid = grid.map(corr, "pred", "real")
    # grid = grid.map(corr, "pred", "real", corre=True)
    grid = grid.add_legend(fontsize=14, title='Area type', frameon=True)

    method_path = ntpath.split(plot_path)[0]
    try_f = ntpath.split(plot_path)[1]

    # grid.savefig(str(method_path) + '/' + str(try_f))

    # save_plot(grid, method_path, try_f)
    save_plot(grid, plot_path, 'pred_real')
    # save_plot(grid, 'C:/Users/Ion/TFM/data/plots/final', 'pred_real')


def regression_plot(study_area_dir, plot_path, sim_path, reg_plots):
    study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
    sim_path = 'C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/two_points'
    plot_path = 'C:/Users/Ion/TFM/data/plots/regression/ols'

    df_lr = pd.read_csv(str(sim_path) + '/linear_regression.csv', sep=",", index_col='area')

    df_sim_N = pd.read_csv(str(sim_path) + '/sim_opt_fs_norm.csv', sep=",", index_col='area')
    df_sim = pd.read_csv(str(sim_path) + '/sim_opt_fs.csv', sep=",", index_col='study_area')
    df_sim_N.columns = pd.MultiIndex.from_tuples([('norm', c) for c in df_sim_N.columns])
    df_sim.columns = pd.MultiIndex.from_tuples([('fs', c) for c in df_sim.columns])

    df_reg = pd.read_csv(str(plot_path) + '/regression_ab.csv', sep=",", index_col='study_area')
    df_fserror = pd.read_csv(str(plot_path) + '/fs_error.csv', sep=",", index_col='study_area')

    df = pd.read_csv(str(study_area_dir) + '/attribute_table_AVG_T.csv', sep=",", index_col='study_area')

    full_df = pd.concat([df, df_sim_N, df_sim, df_lr, df_reg, df_fserror], axis=1, sort=False)
    full_df = full_df.sort_values(by=['network_distance'], ascending=False)

    # df_plots = [['real_a', 'pred_a']]
    pred_list = [['real_a', 'pred_a', 'a'],
                ['real_b', 'pred_b', 'b'],
                ['real_nfs', 'pred_nfs', 'normalized_fs'],
                ['real_fs', 'pred_fs', 'fleet_size']]

    plot_pred_real(full_df, pred_list, plot_path)


def fs_error_plot(data, result_path, area):
    pred, real, error = data
    x = [5, 10, 20, 40, 60, 80, 100]
    numticks = 4
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.rcParams.update({'font.size': 18})

    z = np.polyfit(x, pred, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), color='b', linestyle='dashed', zorder=100)
    ax.scatter(x, pred, color='b', label='Predicted fs', zorder=100)

    z = np.polyfit(x, real, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), color='g', linestyle='dashed', zorder=100)
    ax.scatter(x, real, color='g', label='Real fs', zorder=100)
    ax.set_ylim(0 - max(pred + real) * 0.1, max(pred + real)*1.2)
    # ax.yaxis.set_major_locator(plt.LinearLocator(numticks=numticks)) #THIS
    # '{0:.7g}'.format(float(speed) * 1.609344)
    # ax.set_yticks(np.arange(0, max(pred + real) * 1.2, (max(pred + real) * 1.2)/numticks)) #THIS
    # ax.set_yticks(np.arange(0, max(pred + real) * 1.2, round(((max(pred + real) * 1.2)/numticks), -2)))
    # int(round(123, -2))

    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.f'))

    ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.grid()
    ax.set_xlabel("AV share [%]", fontsize=18)
    ax.set_ylabel("Fleet size", fontsize=18)

    ax2 = ax.twinx()
    ax2.plot(x, error, color="red", marker="o", label='Error', zorder=100)
    ax2.set_ylim(0 - max(error) * 0.1, max(error)*1.2)
    # ax2.yaxis.set_major_locator(plt.LinearLocator(numticks=numticks)) #THIS
    # ax2.set_yticks(np.arange(0, max(error) * 1.2, (max(error) * 1.2)/numticks)) #THIS

    # ax.set_yticks(np.arange(0, max(pred + real) * 1.2, '{0:.2g}'.format((max(pred + real) * 1.2)/numticks)))
    # if max(error) > 1:
    #     rounder = 0
    # elif max(error) > 0.4:
    #     rounder = 2
    # elif max(error) < 0.15:
    #     rounder = 2
    #     # inter =
    # else:
    #     rounder = 2
    # elif max(error) > 1:
    #     formater = '%0.0f'
    # ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) #THIS
    # ax2.set_yticklabels(np.round(np.arange(0, max(error) * 1.2, (max(error) * 1.2)/numticks), rounder))
    # ax.grid()

    # ax2.set_ylabel("Error [ abs (1 - (pred / real) ) ]", fontsize=12)
    ax2.set_ylabel("Error")

    # ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
    # Add legend
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.01, box.width, box.height * 0.9])
    # box = ax2.get_position()
    # ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # fig.legend(loc='upper center', ncol=3, prop={'size': 5}, frameon=True, bbox_to_anchor=(0.5, -0.05))
    # fig.legend(loc='upper center', ncol=1, prop={'size': 15}, frameon=True, bbox_to_anchor=(0.5, 1.0))
    # fig.legend(loc='lower left', ncol=1, prop={'size': 15}, frameon=True, bbox_to_anchor=(0.42, 0.21))
    # fig.legend(loc='best', borderaxespad=0, ncol=1, prop={'size': 13}, frameon=True)
    # plt.title('Predicted fs comparison for: ' + str(area), fontsize=12, fontweight=0)
    # lims = plot_x_vs_y.get_ylim()  # Get min/max of primary y-axis
    # ax2.set_ylim(lims)  # Set min/max of secondary y-axis
    # ax2.grid(None)
    # align_axis(ax, ax2)

    plt.tight_layout()
    # Check if out_path exists or create it
    if not os.path.exists(str(result_path) + '/areas'):
        os.makedirs(str(result_path) + '/areas')

    fig.savefig(str(result_path) + '/areas/' + str(area))

def ols_rf():
    plot_path = 'C:/Users/Ion/TFM/data/plots/regression/ols'

    df_rf = pd.read_csv('C:/Users/Ion/TFM/data/plots/regression/randomForest/rm_less/area_pred_df.csv', sep=',', index_col='study_area')
    comparison_df = pd.read_csv('C:/Users/Ion/TFM/data/plots/regression/randomForest/rm_less/comparison_df.csv', sep=",", index_col='max_attr')
    ix = list(comparison_df.index).index(comparison_df.index[comparison_df['avg_error'] == min(comparison_df['avg_error'])])
    df_rf = df_rf[ix*15:(ix*15)+15]
    # df_rf = df_rf[0:15]

    df_fserror = pd.read_csv(str(plot_path) + '/fs_error.csv', sep=",", index_col='study_area')

    # ols error computing for each area
    ols_error = []
    ols_dict = {}
    for area in df_fserror.index:
        if area == 'plateau':
            continue
        area_error = []
        for column in df_fserror.columns:
            # if 'error' in column:
            if 'error_1.0' in column:
                area_error.append(df_fserror.loc[area, column])
        # print(area, sum(area_error)/len(area_error))
        area_mean = sum(area_error)/len(area_error)
        ols_dict[area] = area_mean
        ols_error.append(sum(area_error)/len(area_error))
    print('ols error', sum(ols_error)/len(ols_error))

    print('------------------')
    # rf error computing for each area
    rf_error = []
    areas = []
    rf_dict = {}
    for area in df_rf.index:
        if 'plateau' in area:
            continue
        areas.append(area)
        area_error = []
        pred_100fs = df_rf.loc[area, 'pred_fs']
        # real_100fs = df_rf.loc[area, 'real_fs']
        # for avshare in [5,10,20,40,60,80,100]:
        for avshare in [100]:
            pred_fs = (pred_100fs / 100) * avshare
            # real_fs = (real_100fs / 100) * avshare
            real_fs = df_fserror.loc[area, 'real_' + str(avshare/100)]
            area_error.append(abs(1-(pred_fs/real_fs)))
        area_mean = sum(area_error)/len(area_error)
        rf_dict[area] = area_mean
        # print(area, area_mean)
        rf_error.append(area_mean)
    print('rf error', sum(rf_error) / len(rf_error))

    cap_areas = {'bern': 'Bern',
                      'chur': 'Chur',
                      'freiburg': 'Freiburg',
                      'frutigen': 'Frutigen',
                      'lausanne': 'Lausanne',
                      'linthal': 'Linthal',
                      'locarno': 'Locarno',
                      'lugano': 'Lugano',
                      'luzern': 'Luzern',
                      'neuchatel': 'Neuchatel',
                      'plateau': 'Plateau',
                      'sion': 'Sion',
                      'stgallen': 'St. Gallen',
                      'zermatt': 'Zermatt',
                      'zurich_kreis': 'Zürich'}
    area_type_dict = {'baden': 'rural',
                      'bern': 'urban',
                        'bern_large': 'urban',
                      'basel': 'urban',
                        'chur': 'mountain',
                        'freiburg': 'rural',
                        'frutigen': 'mountain',
                      'geneve': 'urban',
                      'interlaken': 'mountain',
                      'lausanne': 'urban',
                        'lausanne_lake': 'urban',
                        'linthal': 'mountain',
                        'locarno': 'mountain',
                        'lugano': 'urban',
                        'luzern': 'urban',
                        'neuchatel': 'rural',
                        'plateau': 'rural',
                        'sion': 'mountain',
                        'stgallen': 'rural',
                        'test_area': 'urban',
                      'winterthur': 'rural',
                        'zermatt': 'mountain',
                        'zurich_kreis': 'urban',
                        'zurich_large': 'urban'}

    # comp_df = pd.DataFrame(None, index=[cap_areas[x] for x in areas])
    comp_df = pd.DataFrame(None, index=areas)
    comp_df['ols'] = [x * 100 for x in ols_error]
    comp_df['rf'] = [x * 100 for x in rf_error]


    # Plot i
    # result2 = comp_df.drop(['Plateau']).sort_values(by='ols', ascending=True)
    result = comp_df.sort_values(by='ols', ascending=True)
    ax = result.plot(y=['ols', 'rf'], kind="bar", figsize=(13, 6), label=['Ordinary Least Squares', 'Random Forest Regressor'])

    # result = result2
    plt.hlines(y=sum(result['ols'])/len(result['ols']), xmin=-5, xmax=15, colors='#0089FF', linestyles='dashdot', zorder=100)
    plt.hlines(y=sum(result['ols'])/len(result['ols']), xmin=-5, xmax=15, colors='k', linestyles='dashdot', zorder=0, label='Average prediction error')
    plt.hlines(y=sum(result['rf'])/len(result['rf']), xmin=-5, xmax=15, colors='#FF8900', linestyles='dashdot', zorder=100)
    # plt.hlines(y=68.21, xmin=-5, xmax=15, colors='#FF8900', linestyles='dashdot', zorder=100)

    plt.rcParams.update({'font.size': 16})
    plt.legend(framealpha=None).set_zorder(102)
    plt.grid()
    ax.set_axisbelow(True)
    # ax.set_ylim(0, 67.2)
    ax.set_ylim(0, 100)
    plt.ylabel('Prediction error [%]')

    plt.xticks(rotation=45, ha='right')
    ax2 = ax.twinx()
    # ax2.set_ylim(0,380)
    # ax2.set_ylim(0, 67.2)
    ax2.set_ylim(0, 100)

    # plt.ylabel('Prediction error [%]')
    plt.tight_layout()




    # grouped by area type
    ols_type_avg = []
    rf_type_avg = []
    for area_type in ['urban', 'rural', 'mountain']:
        area_error_ols = []
        area_error_rf = []
        for i in range(len(areas)):
            area = areas[i]
            if area == 'plateau':
                continue
            area_at = area_type_dict[areas[i]]
            if area_at == area_type:
                area_error_ols.append(ols_dict[area])
                area_error_rf.append(rf_dict[area])
        ols_type_avg.append(sum(area_error_ols)/len(area_error_ols))
        rf_type_avg.append(sum(area_error_rf)/len(area_error_rf))

    comp_df2 = pd.DataFrame(None, index=['Urban', 'Rural', 'Mountain'])
    comp_df2['ols'] = [x * 100 for x in ols_type_avg]
    comp_df2['rf'] = [x * 100 for x in rf_type_avg]

    # Plot ii
    # result2 = comp_df.drop(['Plateau']).sort_values(by='ols', ascending=True)
    result = comp_df2.sort_values(by='ols', ascending=True)
    # ax = result.plot(y=['ols', 'rf'], kind="bar", figsize=(10, 6), label=['Ordinary Least Squares', 'Random Forest Regressor'])
    ax = result.plot(y=['ols', 'rf'], kind="bar", figsize=(7, 5), label=['Ordinary Least Squares', 'Random Forest Regressor'])

    # result = result2
    # ols_error.remove(max(ols_error))
    # rf_error.remove(max(rf_error))

    plt.hlines(y=(sum(ols_error)/len(ols_error))*100, xmin=-5, xmax=15, colors='#0089FF', linestyles='dashdot', zorder=100)
    plt.hlines(y=(sum(ols_error)/len(ols_error))*100, xmin=-5, xmax=15, colors='k', linestyles='dashdot', zorder=0, label='Average prediction error')
    plt.hlines(y=(sum(rf_error)/len(rf_error))*100, xmin=-5, xmax=15, colors='#FF8900', linestyles='dashdot', zorder=100)
    # plt.hlines(y=sum(result['ols'])/len(result['ols']), xmin=-5, xmax=15, colors='#0089FF', linestyles='dashdot', zorder=100)
    # plt.hlines(y=sum(result['ols'])/len(result['ols']), xmin=-5, xmax=15, colors='k', linestyles='dashdot', zorder=0, label='Average prediction error')
    # plt.hlines(y=sum(result['rf'])/len(result['rf']), xmin=-5, xmax=15, colors='#FF8900', linestyles='dashdot', zorder=100)


    plt.rcParams.update({'font.size': 19})
    # plt.legend(framealpha=None).set_zorder(102)
    plt.legend(loc='center', framealpha=None, fontsize=17, bbox_to_anchor=(0.5, 1.2)).set_zorder(102)
    plt.grid()
    ax.set_axisbelow(True)
    # ax.set_ylim(0, 67.2)
    ax2.set_ylim(0, 57.2)
    # ax.set_ylim(0, 100)
    plt.ylabel('Prediction error [%]')

    plt.xticks(rotation=0, ha='center')
    # ax2 = ax.twinx()
    # ax2.set_ylim(0,380)
    # ax2.set_ylim(0, 67.2)
    # ax2.set_ylim(0, 57.2)
    # ax2.set_ylim(0, 100)

    # plt.ylabel('Prediction error [%]')
    plt.tight_layout()
    plt.show()


def CarPtusers_fs():
    pred = 'rf'
    # pred = 'ols'
    fs = ('norm', '1.0')
    # fs = ('fs', '1.0')
    # attr = 'trips'
    # attr = 'CarPt_users'
    attr = 'real_1.0'
    # attr = 'node_load_centrality'
    try:
        full_df = full_df.drop(['winterthur', 'interlaken', 'geneve', 'baden', 'basel', 'plateau'])
    except:
        pass
    # g1 = (full_df[full_df['area_type'] == 'rural'][attr], full_df[full_df['area_type'] == 'rural'][fs])
    # g2 = (full_df[full_df['area_type'] == 'mountain'][attr], full_df[full_df['area_type'] == 'mountain'][fs])
    # g3 = (full_df[full_df['area_type'] == 'urban'][attr], full_df[full_df['area_type'] == 'urban'][fs])
    area_type = []
    for ix in result.index:
        area_type.append(area_type_dict[ix])
    result['area_type'] = area_type
    # g1 = (full_df[full_df['area_type'] == 'rural'][attr], result[result['area_type'] == 'rural'][pred])
    # g2 = (full_df[full_df['area_type'] == 'mountain'][attr], result[result['area_type'] == 'mountain'][pred])
    # g3 = (full_df[full_df['area_type'] == 'urban'][attr], result[result['area_type'] == 'urban'][pred])
    g1 = (full_df[attr], result['ols'])
    g2 = (full_df[attr], result['rf'])
    # g2 = (full_df[full_df['area_type'] == 'mountain'][attr], result[result['area_type'] == 'mountain'][pred])
    # g1 = (full_df[full_df['area_type'] == 'rural']['node_load_centrality'], full_df[full_df['area_type'] == 'rural'][fs])
    # g2 = (full_df[full_df['area_type'] == 'mountain']['node_load_centrality'], full_df[full_df['area_type'] == 'mountain'][fs])
    # g3 = (full_df[full_df['area_type'] == 'urban']['node_load_centrality'], full_df[full_df['area_type'] == 'urban'][fs])


    # data = (g1, g2, g3)
    data = (g1, g2)
    colors = ("blue", "#FF7400")
    groups = ("Ordinary Least Squares", "Random Forest Regressor")
    # colors = ("red", "green", "blue")
    # groups = ("rural", "mountain", "urban")

    # Create plot
    # fig = plt.figure(figsize=(6,4))
    fig = plt.figure(figsize=(7,5))
    plt.rcParams.update({'font.size': 19})
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')
    ax.set_xlim(50, 20000)
    ax.set_ylim(0, 100)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=50, label=group, zorder=100)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.title('Matplot scatter plot')
    plt.xlabel('CarPt_users')
    plt.xlabel('Fleet size')
    # plt.xlabel('Trips')
    # plt.xlabel(attr)
    # plt.xlabel('Load centrality')
    # plt.ylabel('Normalized fleet size')
    plt.ylabel('Prediction error [%]')
    # plt.ylabel('Full fleet size')
    # plt.legend(loc='left center', fontsize=14, bbox_to_anchor=(1.05, 0.6))
    plt.legend(loc='center', fontsize=17, bbox_to_anchor=(0.5, 1.2))
    plt.grid()
    plt.tight_layout()
    plt.show()


def static_demand():
    fig = plt.figure()
    plt.rcParams.update({'font.size': 16})
    ax = plt.axes()
    x = np.linspace(1000, 2000, 10)
    y = [11.6, 8.7, 7, 5.75, 4.6, 3.8, 3, 2.3, 1.9, 1.6]
    x = np.linspace(1000, 2000, 10)
    y = [11.6, 8.7, 7, 5.75, 4.6, 3.8, 3, 2.3, 1.9, 1.6]
    ax.plot(0, 0, marker='o')
    # ax.hlines(y=5, xmin=0, xmax=1400, colors='r', linestyles='dashdot', zorder=100, label='Threshold')
    # ax.vlines(x=1400, ymin=0, ymax=5, colors='r', linestyles='dashdot', zorder=100, label='Threshold')
    ax.set_ylim(0, 12)
    ax.set_xlim(700, 2050)
    plt.yticks([1,3,5,7,9,11], ['1','3','5','7','9','11'])
    # ax.get_xticklabels()[4].set_color("red")
    plt.xlabel('Fleet size')
    plt.ylabel('Average waiting time [min]')
    plt.tight_layout()

# -----------------------------------------------------------------------------
# SIMULATION RESULTS PLOTS
# -----------------------------------------------------------------------------

# This creates plot for each area (wt vs fs) and opt fs csv (fs, and fs_N)
fit_func = 'best'     # ['exp', 'power', 'best', 'two_points']

# plot_sim_results(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#                  plot_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/' + str(fit_func),
#                  fit_func=fit_func,
#                  area=None)
# #
# # # This creates for a defined fs or fs_N csv, the corresponding plot fs vs avshare value, with a regression (linear, exp)
# plot_fs_avs_setup(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#                   plot_path='C:/Users/Ion/TFM/data/plots/sim_plots',
#                   regression='linear',       # ['linear' (saves a b as attributes), 'exp', 'power', 'quadratic', 'best']
#                   fit=fit_func)              # ['exp', 'power', 'best', 'two_points'] any already computed fit foldername of wt/fs


# correlation_matrix(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#                    sim_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/' + str(fit_func),
#                    fit=fit_func)




# -----------------------------------------------------------------------------
# ATTRIBUTE PLOTS
# -----------------------------------------------------------------------------

# # Different facet plots:
df_plots = [
#     [['network_distance', 'efficiency', 'node_straightness', 'eccentricity', 'avg_shortest_path_duration', 'area_type'], 'random'],
#     [['network_distance', 'node_d_km', 'edge_d_km', 'intersection_d_km', 'street_d_km', 'area_type'], 'densities'],
    [['network_distance', 'node_d_km', 'street_d_km', 'area_type'], 'densities'],
    [['CarPt_users', 'node_load_centrality', 'node_straightness', 'area_type'], 'other']
#     [['network_distance', 'population', 'trips', 'area', 'circuity_avg', 'area_type'], 'dimensionless'],
#     [['network_distance', 'avg_degree_connectivity', 'avg_edge_density', 'degree_centrality', 'avg_neighbor_degree', 'area_type'], 'degrees'],
#     [['network_distance', 'clustering*', 'node_betweenness*', 'edge_betweenness', 'street_d_km', 'area_type'], 'dicts'],
#     [['network_distance', 'btw_home_trip_production', 'btw_empl_trip_generation', 'btw_acc_trip_generation', 'btw_acc_trip_production', 'area_type'], 'btw_acc']
#     # [['network_distance', 'avg_degree_connectivity', 'intersection_d_km', 'avg_neighbor_degree', 'area_type'], 'btw_acc']
#     [['population_gini', 'population','population_density', 'a', 'b', '1.0', 'area_type'], 'pop_gini']
]
# setup_gridplot(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#                plot_path='C:/Users/Ion/TFM/data/plots/attribute_plots/facetPlot/simpler',
#                df_plots=df_plots)


# regression_plot(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#                 plot_path='C:/Users/Ion/TFM/data/plots/regression/ols',
#                 sim_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/two_points',
#                 reg_plots=[[['real_a', 'pred_a','real_b', 'pred_b', 'real_nfs', 'pred_nfs', 'real_fs', 'pred_fs'], 'reg_results']])

# plot predefined attributes:
# attr_list = ['node_d_km', 'edge_d_km', 'CarPt_users', 'trips',
#              'network_distance', 'circuity_avg', 'avg_degree', 'avg_neighbor_degree',
#              'degree_centrality', 'node_straightness', 'node_closeness_time*', 'efficiency',
#              'btw_home_trip_production', 'btw_empl_trip_generation', 'btw_acc_trip_generation', 'btw_acc_trip_production']
# attr_list = ['avg_degree', 'node_d_km']
# attr_list = ['node_closeness_time*']

# plot_fs_attr_setup(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#                    sim_path='C:/Users/Ion/TFM/data/plots/sim_results/wt_fs/' + str(fit_func),
#                    plot_path='C:/Users/Ion/TFM/data/plots/regression',
#                    attr_list=attr_list)







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




