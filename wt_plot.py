import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from scipy.optimize import curve_fit
import ntpath

# column='0.4'
def tend_curve(df, column):
    def exponential(x, a, k, b):
        return a * np.exp(-k * x) + b
    def quadratic(x, a, k, b):
        return a * x** 2 + k * x + b

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
        # print('quadratic better')
    else:
        model_func = exponential
        # print('exponential better')


    opt, pcov = curve_fit(model_func, x, y, p0=(1., 1.e-5, 1.), maxfev=10000)
    a, k, b = opt

    # test result
    x2 = np.linspace(min(x), max(x), 1000)
    y2 = model_func(x2, a, k, b)
    return x2, y2

    # fig, ax = plt.subplots()
    # ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.9f x} %+.3f$' % (a, k, b))
    # # ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.9f x^2 + %.3f x + %+.3f$' % (a, k, b))
    # ax.plot(x, y, 'bo', label='data with noise')
    # ax.legend(loc='best')
    # plt.title('Distribution of trips in study area')
    # plt.xlabel('trip length (in seconds)')
    # plt.ylabel('count (normalized)')
    # plt.show()

# area_path = r'C:\Users\Ion\TFM\data\study_areas/locarno'
def plot_fc(area_path):
    area = ntpath.split(area_path)[1]
    df_name = 'avg_df'
    df = pd.read_csv(str(area_path) + '/simulations/' + str(df_name) + '.csv', sep=",", index_col='fleet_size')

    # style
    plt.figure(figsize=(15, 8))
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('tab10')
    # multiple line plot
    num = 0
    for column in df:
        x = df.index
        y = df[column]
        xx, yy = tend_curve(df, column)
        # plt.plot(x, y, 'o', xx, yy, color=palette(num), label=column)
        # plt.plot(df.index, df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
        plt.scatter(x, y, color=palette(num), label=column)
        plt.plot(xx, yy, color=palette(num), linestyle='dashed')
        num += 1

    # Add legend
    plt.legend(loc=0, ncol=1, prop={'size': 14}, frameon=True, title='AV share value:')

    # Define axis
    left, right = plt.xlim()
    bottom, top = plt.ylim()
    ax = plt.axes()
    ax.set_ylim(125, 600)
    ax.set_xlim(left, right)

    # Add hline at wt 300 s
    plt.hlines(y=300, xmin=left, xmax=right, colors='r', linestyles='dashdot', zorder=100, label='Threshold')
    plt.text(2000, 310, 'Threshold ', ha='right', va='center', fontsize=10, color='r')
    # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(formatter))

    # Add titles
    plt.title('AV simulation results for: ' + str(area) + ' (10pct swiss census)', fontsize=16, fontweight=0,
              color='orange')
    plt.xlabel("AVs Fleet Size", fontsize=10)
    plt.ylabel("Waiting Time (s)", fontsize=10)

    plt.savefig(str(area_path) + '/simulations/' + str(df_name) + '.png')

    path = ntpath.split(area_path)[0]
    data_path = ntpath.split(path)[0]
    plt.savefig(str(data_path) + '/plots/sim_plots/' + str(area) + '.png')


# Parsing command line arguments
# parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
# parser.add_argument('--area-path', dest="area_path", help='path to simulation_output folder')
# args = parser.parse_args()
#
# area_path = args.area_path.split('\r')[0]

# plot_fc(area_path)





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