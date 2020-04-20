import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg')
# mpl.use('TkAgg')
import numpy as np
import pandas as pd
import argparse
from scipy.optimize import curve_fit
import ntpath
import os
import datetime
import math
import cmath

# column='0.4'
def tend_curve(df, column):
    def exponential(x, a, k, b):
        return a * np.exp(-k * x) + b
    def inv_exponential(y, a, k, b):
        return (1 / k) * math.log(a / (y - b))

    def quadratic(x, a, k, b):
        return a * x** 2 + k * x + b
    def inv_quadratic(y, a, k, b):
        # calculate the discriminant
        c = b - y
        d = (k ** 2) - (4 * a * c)

        # find two solutions
        sol1 = (-k - cmath.sqrt(d)) / (2 * a)
        sol2 = (-k + cmath.sqrt(d)) / (2 * a)

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

    # curve fit
    y = df[column]
    nan_elems = y.isnull()
    y = y[~nan_elems]
    for ix, wt in y.iteritems():
        if wt > 800 or wt < 100:
            y = y.drop(labels=[ix])
    x = y.index

    exp_error = error_f(x, y, exponential)
    quad_error = error_f(x, y, quadratic)

    if exp_error > quad_error:
        model_func = quadratic
        inv_func = inv_quadratic
        print('quadratic better')
    else:
        model_func = exponential
        inv_func = inv_exponential
        print('exponential better')


    opt, pcov = curve_fit(model_func, x, y, p0=(1., 1.e-5, 1.), maxfev=10000)
    a, k, b = opt
    # print(a, k, b)
    # test result
    x2 = np.linspace(min(x), max(x), 1000)
    y2 = model_func(x2, a, k, b)

    # Get threshold value of the fitted curve for a threshold of 300 wt
    threshold = math.ceil(inv_func(300, a, k, b))
    print('for: ' + str(column) + ' fleet size of: ' + str(threshold))

    return x2, y2, threshold

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
    plot_path = str(data_path) + '/plots/sim_plots/try'

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
    threshold_dict = {}
    for column in df:
        x = df.index
        y = df[column]
        xx, yy, threshold = tend_curve(df, column)
        df_update(study_area_path, column, threshold, area)
        # plt.plot(x, y, 'o', xx, yy, color=palette(num), label=column)
        # plt.plot(df.index, df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
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
    plt.xlabel("AVs Fleet Size", fontsize=14)
    plt.ylabel("Waiting Time (s)", fontsize=14)

    # plt.savefig(str(area_path) + '/simulations/' + str(df_name) + '.png')
    plt.savefig(str(plot_path) + '/' + str(area) + '.png')


# Parsing command line arguments
# parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
# parser.add_argument('--area-path', dest="area_path", help='path to simulation_output folder')
# args = parser.parse_args()
#
# area_path = args.area_path.split('\r')[0]

# plot_fc(area_path)

study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'

study_area_list = list(os.walk(study_area_dir))[0][1]
# study_area_list = ['bern']

for area in study_area_list:
    if area == 'test_area':
        continue
    area_path = str(study_area_dir) + '/' + str(area)
    plot_fc(area_path)








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