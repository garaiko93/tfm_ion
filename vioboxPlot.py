# python script created from: https://github.com/ciortanmadalina/modality_tests/blob/master/violinboxplot_hybrid_axes.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# plt.ion()
# plt.show()

# Let's start by choosing a function (sin) which will generate the dataset used to demonstrate the hybrid scales concept.
# x = np.arange(-80,80, 0.1)
# y = np.sin(x)
# plt.title('Linear scale plot of a sinusoid')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(x, y);

# If we treat this dataset as a blackbox, a data scientist may want, for any number of reasons to have a dynamic resolution of the plot by using different scales on different intervals with a minimal effort. For instance, he may want to visualize:
#
# 1<= y <= 0.5 using a linear scale
# 0.1 <= y <= 0.5 using a log scale
# -1<= y <= 0.1 using a linear scale
# The first naive solution is to create 3 different plots with the chosen axis scales on chosen intervals.
# In this post we will investigate the capabilities of matplotlib to make appear the original plot under different scales, thus providing a unified visualization.
#
# There are 2 approaches we will present:
#
# Using an axis divider
# Using Grid spec

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Matplotlib's function make_axes_locatable allow us to append a new axis to a given axis. In the example below, a log axis is created from the original linear axis.
# By setting arbitrary y limits we control what part of the plot is being rendered and we can create the impression of plot continuity.
# Sharedx parameter allows sharing the same x axis and prevents the x tick labels from being rerendered.

# linearAxis = plt.gca()
# linearAxis.plot(x, y)
# linearAxis.set_ylim((0.5, 1))
#
# divider = make_axes_locatable(linearAxis)
# logAxis = divider.append_axes("bottom", size=1, pad=0.02, sharex=linearAxis)
# logAxis.plot(x, y)
# logAxis.set_yscale('log')
# logAxis.set_ylim((0.01, 0.5));

# We can use append axes on a given input axis in 4 potential location (top/ bottom/ up/ down).
# The code below illustrates chaining 2 axes, on top and on bottom.
# logAxis = plt.gca()
# logAxis.plot(x, y)
# logAxis.set_yscale('log')
# logAxis.set_ylim((0.01, 0.5))
#
# divider = make_axes_locatable(logAxis)
# linearAxis = divider.append_axes("top", size=1, pad=0.02, sharex=logAxis)
# linearAxis.plot(x, y)
# linearAxis.set_ylim((0.5, 1))
# linearAxis.set_xscale('linear')
# linearAxis.set_title('Plot split in 3 scales: linear: [0.5, 1], log: [0.01, 0.5], linear: [-1, 0.01]');
#
# linearAxis1 = divider.append_axes("bottom", size=1, pad=0.02, sharex=logAxis)
# linearAxis1.plot(x, y)
# linearAxis1.set_yscale('linear')
# linearAxis1.set_ylim((-1, 0.01));

# GridSpec implementation
# Another option is to use matplotlib's GridSpec which provides more flexibility in terms of sizing the components and usage.
# We can define upfront the number of suplots, their relative sizes (height_ratios), the distance between subplots (hspace).
# Once the independent axis have been created, we can set the scales and the desired limits.
import matplotlib.gridspec as grd
# gs = grd.GridSpec(3, 1, wspace=0.01, hspace=0.05, height_ratios=[0.33, 0.33, 0.33])
#
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1])
# ax3 = plt.subplot(gs[2])
# ax1.set_xticks([])
# ax2.set_xticks([])
#
# ax1.plot(x, y)
# ax1.set_yscale('linear')
# ax1.set_ylim((0.5, 1))
#
# ax2.plot(x, y)
# ax2.set_yscale('log')
# ax2.set_ylim((0.01, 0.5))
#
# ax3.plot(x, y)
# ax3.set_yscale('linear')
# ax3.set_ylim((-1, 0.01));

# Custom violinbloxplot
# Let's start by generating a few data distributions reflecting multiple scenarios:
#
# unimodal data reflecting a gaussian distribution
# a combination of gaussian data with outliers
# a dataset with multiple (7 such distributions) illustrating a comparative visualization of input distributions
# a dataframe to be grouped by one or multiple comumns to illustrate the compartive data distribution

# data1 = [np.round(np.random.normal(10, 0.4, 50), 2)]
# data1SharpEnd = [[e for e in data1[0] if e > 9.9]]
# data1Spread = [
#     np.concatenate([
#         np.round(np.random.normal(10, 0.2, 1000), 2),
#         np.round(np.random.normal(80, 0.3, 5), 2)
#         ])
#     ]
#
# data2 = [
#     np.concatenate([
#         np.round(np.random.normal(10, std/10, 1000), 2),
#         np.round(np.random.normal(80, std, np.random.randint(0, 24) * std), 2) ])
#             for std in range(1, 7)
#         ]
# labels7 = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Based on one of the existing datasets, we can define a dataframe:

# df = pd.DataFrame()
# df['values'] = data1Spread[0]
# df['col1'] = np.random.choice(['A', 'B'], df.shape[0])
# df['col2'] = np.random.choice(['C', 'D'], df.shape[0])
# df.head()

# In order to better understand the underlying data distribution, let's create a plotting function which leverages both boxplots and violinplots:
def plotDistributions(inputData, title):
    """
    This method plots inputData with:
    - matplotlib boxplot
    - matplotlib violinplot
    - seaborn violinplot
    """
    globalMax = np.max(np.concatenate(inputData))
    globalMin = np.min(np.concatenate(inputData))

    plt.figure(figsize=(14, 4))
    plt.suptitle(title)

    plt.subplot(121)
    plt.grid()
    plt.title('Matplotlib boxplot')
    plt.boxplot(inputData, vert=False);
    plt.axvline(x=globalMax, c='red', label='Global max', alpha=0.5)
    plt.axvline(x=globalMin, c='red', label='Global min', alpha=0.5)
    plt.legend()

    plt.subplot(122)
    plt.grid()
    plt.title('Matplotlib violinplot')
    plt.violinplot(inputData, vert=False, showmeans=False, showmedians=True, showextrema=True);
    plt.axvline(x=globalMax, c='red', label='Global max', alpha=0.5)
    plt.axvline(x=globalMin, c='red', label='Global min', alpha=0.5)
    plt.legend()

# plotDistributions(data1, 'Distribution of data1')
# plotDistributions(data1Spread, 'Distribution of data1Spread')
# plotDistributions(data1SharpEnd, 'Distribution of data1SharpEnd');
# plotDistributions(data2, 'Distribution of data1SharpEnd');
#
# # We can visualize the dataframe using seaborn
# sns.violinplot(x = 'values', y='col1', data = df)
# plt.figure()
# sns.violinplot(x = 'values', y='col2', data = df)

# However, seaborn expects to indicate as y only one column which will be used in a group by to aggregate the results.
# If we want to aggregate based on a combination of multiple features, we have to do it prior to calling the plotting function.
#
# What some drawbacks we can identify in the above plots?
#
# it would be nice to have the combined resolution of boxplots and violin plots in one graph. Seaborn offers through
# the inner parameter a way to incorporate a boxplot but its customisation possibilities are limited.
# as shown in the second graphic, if we are dealing with a distribution with far outliers, the overall visualization
# looses the details at the extremes. What if we could used the examples discussed in the first section to create a customized unified view with arbitrary scales on target intervals?
# Some other points to consider are:
#
# how can we enrich plots with custom annotations indicating for instance, for each dataset the number of points and
# other arbitrary measures, such as the mode?
# could be provide a hyperparameter which could remove from the visualisation alltogether the points
# we consider outliers?
# If we start with this last point, we can come up with a method that removes all points father away than
# a given number of standard deviations (by default 3).

def removeOutliers(data, thresholdStd = 3):
    """
    This method returns all values which are farther away
    than thresholdStd standard deviationa
    """
    noOutliers=[]
    mean = np.mean(data)
    std =np.std(data)
    if std == 0:
        return data

    for y in data:
        z_score= (y - mean)/std
        if np.abs(z_score) <= thresholdStd:
            noOutliers.append(y)
    return noOutliers

# Implementation of violinboxplot
def violinboxplot(data, x=None, y=None, labels=None, outliers=True,
                  title=None, showModes=True, showCounts=True,
                  logPercentile=None, ax=None, xtitle=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import ScalarFormatter, NullFormatter

    """
    This method takes as input data as dataframe or list of lists.
    When data is a dataframe, we are expecting x (measured feature) and y(group by criteria
    which can be one or multiple columns) to be provided.
    When data is a list we are expending label values

    Credit to https://matplotlib.org/gallery/statistics/customized_violin.html
    :param data: if data is not a dataframe, it is expected to be a list of lists, where each nested 
    list will be plotted separately
    :param x = None: if data is a dataframe, x represents the column with target variables to plot

    :param y = None: if data is a dataframe, y is expected to be the list of columns used to make
    a groupby on.
    :param labels = None: if data is a datframe, y will be deduced from the index. Otherwise this vector
    will hold the y labels for the collection of datasets rendered as violinboxplot

    :param outliers = True: Keeps all input datapoints when outliers = True. If outliers = False
    removes outliers (points futher away than 3 standard deviations)

    :param title = 'violinboxplot': The figure's title

    :param showModes = True: for each distribution annotates the plot with the mode values

    :param showCounts = True: for each distribution annotates the plot with the number of observations (counts)

    :param logPercentile = None: When percentile is a value between 0 and 1 splits the axis in 2 parts, the first 
    being in linear scale and the second logscale. The split point is the percentile value indicated by logPercentile

    :param ax = None: Ax to be used for plotting
    :return 

    """

    # Helper functions
    def basicPlot(ax, data, labels, xlim1=None, xlim2=None):
        """
        The actual plotting function. This method will be called a second time
        if logPercentile parameter is set, providing a secondary axis
        """

        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, vert=False);
        for i in range(len(parts['bodies'])):
            pc = parts['bodies'][i]
            label = labels[i]
            # print(label, pc)

            urban = ['luzern', 'bern', 'zurich_kreis', 'lausanne', 'lugano', 'stgallen']
            rural = ['freiburg', 'neuchatel', 'plateau']
            mountain = ['chur', 'sion', 'linthal', 'frutigen',  'zermatt', 'locarno']

            if label in urban:
                pc.set_facecolor('#6290db')
            elif label in rural:
                pc.set_facecolor('#cf1dab')
            elif label in mountain:
                pc.set_facecolor('#36d957')

            # pc.set_facecolor('#6290db')
            pc.set_edgecolor('#4a6591')
            pc.set_alpha(0.8)

        # for pc in parts['bodies']:
        #     pc.set_facecolor('#6290db')
        #     pc.set_edgecolor('#4a6591')
        #     pc.set_alpha(0.8)
        ax.boxplot(data, vert=False);

        if xlim1 is not None and xlim2 is not None:
            ax.set_xlim((xlim1, xlim2))
            xlimits = True
        else:
            xlimits = False
        plt.grid()
        for i in range(len(data)):
            if showModes:
                modes = stats.mode(data[i])
                for j, mode in enumerate(modes[0]):
                    if xlimits == False or (mode <= xlim2 and mode >= xlim1):
                        plt.scatter(mode, i + 1, marker='d', color='green', s=30, zorder=3)
                        ax.text(mode, i + 1.3, f'#{modes[1][j]}: {np.round(mode, 5)}', color='green')
            if showCounts:
                # show counts at the end of each line
                globalMax = np.max(np.concatenate(data))
                if xlimits == False or (globalMax <= xlim2 and globalMax >= xlim1):
                    ax.text(globalMax, i + 1, f'#{len(data[i])}', color='black')

    def prepareData(data, labels):
        if isinstance(data, pd.DataFrame):
            dd = data.copy()
            dd.loc[:, 'target'] = dd[y].apply(lambda x: '_'.join(x), axis=1).values
            dd = dd.dropna().groupby('target').agg({x: [list, 'size']}).sort_values(by=(x, 'size'), ascending=True)
            dd.columns = [x, 'count']
            data = dd[x].values
            labels = dd.index.values

        if outliers == False:
            data = [removeOutliers(v) for v in data]

        return data, labels

    def plotData(data, ax, logPercentile, labels):
        """
        This method handles the split between log percentile and uniaxis case.
        For the case of logpercentile we need to send to the plot data function
        the xlimits in order to handle the axis specific annotations.
        """
        if logPercentile is not None:
            globalMin = np.min(np.concatenate(data))
            globalMax = np.max(np.concatenate(data))
            globalMax = globalMax + globalMax * 0.1
            split = np.percentile(np.concatenate(data), 90)
            basicPlot(ax, data, labels, globalMin, split)

            divider = make_axes_locatable(ax)  # create log axis
            axLog = divider.append_axes("right", size=3, pad=0.02)
            axLog.set_xscale('log')
            basicPlot(axLog, data, labels, split, globalMax)
            axLog.set_yticks([])  # no annoations, we use the ones from the linear axis
            axLog.xaxis.set_minor_formatter(ScalarFormatter())
            axLog.xaxis.set_major_formatter(NullFormatter())

        else:
            basicPlot(ax, data, labels)

    ## Logic starts here
    data, labels = prepareData(data, labels)
    if ax == None:
        plt.figure(figsize=(16, len(data) * 0.5))
        # plt.figure(figsize=(16, len(data)*1.5))
        ax = plt.gca()
    plotData(data, ax, logPercentile, labels)
    if labels is not None:  # set labels
        ax.set_yticklabels(labels)
    plt.title(title) if logPercentile is None else plt.suptitle(title)
    if xtitle:
        plt.xlabel(xtitle)
    plt.tight_layout()
    sns.despine()


# Let's see plot function on the datasets presented above:
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(data1, showModes=True, ax = ax, logPercentile=None, labels=labels7, showCounts=True, outliers=True,
#               title="Data1 no logpercentile")
#
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(data1Spread, showModes=True, ax = ax, logPercentile=None, labels=labels7, showCounts=True, outliers=True,
#               title="data1Spread no logpercentile with outliers")
#
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(data1Spread, showModes=True, ax = ax, logPercentile=None, labels=labels7, showCounts=True, outliers=False,
#               title="data1Spread no logpercentile without outliers")
#
#
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(data1Spread, showModes=True, ax = ax, logPercentile=0.9, labels=labels7, showCounts=True, outliers=True,
#               title="data1Spread logpercentile = 0.9 with outliers")
#
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(data2, showModes=True, ax = ax, logPercentile=None, labels=labels7, showCounts=True,
#               outliers=True, title="data2 with outliers")
#
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(data2, showModes=True, ax = ax, logPercentile=0.9, labels=labels7, showCounts=True,
#               outliers=True, title="data2 with logpercentile = 0.9")
#
# # Violinplot applied to dataframe:
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(df, x = 'values', y = ['col1'],
#               showModes=True, ax = ax, logPercentile=0.9, labels=labels7, showCounts=True,
#              title="Dataframe grouped by col1")
#
#
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(df, x = 'values', y = ['col2'],
#               showModes=True, ax = ax, logPercentile=0.9, labels=labels7, showCounts=True,
#              title="Dataframe grouped by col2")
#
#
#
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# violinboxplot(df, x = 'values', y = ['col1', 'col2'],
#               showModes=True, ax = ax, logPercentile=0.9, labels=labels7, showCounts=True,
#              title="Dataframe grouped by col1 and col2")