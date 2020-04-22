import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg')
# mpl.use('TkAgg')
import numpy as np
import pandas as pd
# import argparse
from scipy.optimize import curve_fit
import ntpath
import os
import datetime

from plot_attributes import save_plot

# column='0.4'
# def tend_curve(df, column):
def tend_curve(x, y, column=None):
    def exponential(x, a, k, b):
        return a * np.exp(-k * x) + b
    def inv_exponential(y, a, k, b):
        return (1 / k) * np.log(a / (y - b))

    def quadratic(x, a, k, b):
        return a * x** 2 + k * x + b
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

# study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
#
# study_area_list = list(os.walk(study_area_dir))[0][1]
# study_area_list = ['bern']

# for area in study_area_list:
#     if area == 'test_area':
#         continue
#     area_path = str(study_area_dir) + '/' + str(area)
#     plot_fc(area_path)


# -----------------------------------------------------------------------------
# PLOT OPT_FS
# -----------------------------------------------------------------------------
def plot_opt_fs(df):
    print(datetime.datetime.now(), 'Creating plot ...')
    # style
    plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 14})
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    NUM_COLORS = len(df)
    # palette = plt.get_cmap('tab20')
    # palette = plt.get_cmap('terrain')
    palette = plt.get_cmap('gist_rainbow')
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
    plt.ylabel("Optimal AVs Fleet Size for 300 sec Waiting Time", fontsize=16)

    save_plot(plt, str(data_path) + '/plots/sim_plots/opt_fs_plot/', 'opt_fs_plot')
    # plt.savefig(str(data_path) + '/plots/sim_plots/opt_fs_plot/opt_fs_plot.png')

print('plot starts')
study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'
data_path = ntpath.split(study_area_dir)[0]
df = pd.read_csv(str(study_area_dir) + '/' + 'sim_threshold.csv', sep=",", index_col='area')
df = df.sort_values(by=['1.0'], ascending=False)
dfs = [df, df[df['1.0'] > 3000], df[df['1.0'] <= 3000]]
# dfs = [df]
for df in dfs:
    plot_opt_fs(df)







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