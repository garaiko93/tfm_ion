import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.patches as mpatches

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt



def prepare_df(study_area_dir, sim_path, out_path, pred_var, drop_attr=None, select_attr=None):
    # Check if out_path exists or create it
    if not os.path.exists(str(out_path)):
        os.makedirs(str(out_path))
        # print('Directory created')
    # else:
        # print('Directory exists.')

    df_lr = pd.read_csv(str(sim_path) + '/linear_regression.csv', sep=",", index_col='area')

    df_sim_N = pd.read_csv(str(sim_path) + '/sim_opt_fs_norm.csv', sep=",", index_col='area')
    df_sim = pd.read_csv(str(sim_path) + '/sim_opt_fs.csv', sep=",", index_col='area')
    df_sim_N.columns = pd.MultiIndex.from_tuples([('norm', c) for c in df_sim_N.columns])
    df_sim.columns = pd.MultiIndex.from_tuples([('fs', c) for c in df_sim.columns])

    df_attr = pd.read_csv(str(study_area_dir) + '/attribute_table_AVG_T.csv', sep=",", index_col='study_area')

    full_df = pd.concat([df_attr, df_sim_N, df_sim, df_lr], axis=1, sort=False)
    full_df = full_df.sort_values(by=['network_distance'], ascending=False)

    if pred_var[0] == 'norm':
        df = pd.concat([df_attr, df_sim_N[pred_var]], axis=1, sort=False)
    # elif pred_var[0] == 'fs':
    else:
        df = pd.concat([df_attr, df_sim[pred_var]], axis=1, sort=False)

    # Replace area_type ('mountain', 'urban', 'rural') for numerical features
    df = pd.get_dummies(df)

    if drop_attr is not None and len(drop_attr) > 0:
        for attr in drop_attr:
            df = df.drop(attr, axis=1)
    if select_attr is not None:
        df = df[select_attr]


    # df = pd.concat([df_attr.drop(['node_load_centrality', 'edge_betweenness'], axis=1), df_sim_N[pred_var]], axis=1, sort=False)
    # df = df.drop('plateau')
    df = df.drop(['baden', 'interlaken', 'geneve', 'basel', 'winterthur'])

    # Check df
    check_df = pd.concat([df_attr['CarPt_users'], df_sim_N, df_sim, df_lr], axis=1, sort=False)
    check_df = check_df.drop(['baden', 'interlaken', 'geneve', 'basel', 'winterthur'])
    return df, check_df


def randomForest(df, test_area, n_estimators):
    full_train = df.drop(test_area)
    full_test = df.loc[test_area, :]
    feature_list = list(df.drop(pred_var, axis=1).columns)

    train_features = np.array(full_train.drop(pred_var, axis=1))
    train_labels = np.array(full_train[pred_var])

    test_features = np.array(full_test.drop(pred_var)).reshape(1, -1)
    test_labels = np.array(full_test[pred_var])

    # # Split the data into training and testing sets
    # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=(1/len(features)), random_state=42)
    # # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)


    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=100)
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # path = rf.decision_path(test_features)
    # errors = round(np.mean(abs(1 - (predictions / test_labels)), 4)
    error = round(np.mean(abs(1 - (predictions / test_labels))), 4)
    # print(test_area, 'Error:', error, predictions, test_labels)


    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    feature_importances_dict = {}
    for pair in feature_importances:
        feature_importances_dict[pair[0]] = pair[1]

    # i = 0
    # if len(feature_importances) < 5:
    #     limit = len(feature_importances)
    # else:
    #     limit = 10
    # while i < limit:
    #     print('Variable: ', feature_importances[i][0], 'Importance: ', feature_importances[i][1])
    #     i +=1
    # print('----------------------------------')
    return error, feature_importances_dict, predictions[0], test_labels

    # Print out the feature and importances
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    # i = 0
    # if len(feature_importances) < 10:
    #     limit = len(feature_importances)
    # else:
    #     limit = 10
    # while i < limit:
    #     print('Variable: ', feature_importances[i][0], 'Importance: ', feature_importances[i][1])
    #     i +=1
    # return round(np.mean(errors), 4)

    # list of x locations for plotting
    # x_values = list(range(len(importances)))
    # # Make a bar chart
    # plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
    # # Tick labels for x axis
    # plt.xticks(x_values, feature_list, rotation='vertical')
    # # Axis labels and title
    # plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


    # out_file = 'C:/Users/Ion/TFM/data/plots/regression/randomForest'
    # # Pull out one tree from the forest
    # tree = rf.estimators_[5]
    # # Export the image to a dot file
    # export_graphviz(tree, out_file=out_file + '/tree.dot', feature_names=feature_list, rounded=True, precision=4)
    # # Use dot file to create a graph
    # (graph, ) = pydot.graph_from_dot_file(out_file + '/tree.dot')
    # # Write graph to a png file
    # graph.write_png(out_file + '/tree.png')






# full_df = prepare_df(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#                      sim_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/two_points',
#                      out_path='C:/Users/Ion/TFM/data/plots/regression/ols/try8',
#                      pred_var=('norm', '1.0'),
#                      drop_attr=None,
#                      select_attr=None)

# attr_sel = ['population_density', 'CarPt_users', 'efficiency', 'btw_acc_trip_generation', 'node_straightness']
# attr_sel = ['CarPt_users', 'population_density', 'area']

# ----------------------------------------------------------------------------------------------
# PLOT i: comparison of most important variables fs/nfs
# ----------------------------------------------------------------------------------------------
comparison_df = None
variables = [('norm', '1.0'), ('fs', '1.0')]
for i in range(len(variables)):
    # Prepare data to build model
    pred_var = variables[i]
    df, check_df = prepare_df(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
                              sim_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/two_points',
                              out_path='C:/Users/Ion/TFM/data/plots/regression/randomForest',
                              pred_var=pred_var,
                              drop_attr=None,
                              select_attr=None)


    # Iterate over areas to predict on them
    fi_df = pd.DataFrame(None, columns=df.columns)
    for test_area in df.index:
        error, feature_imp, pred, real = randomForest(df=df,
                                                      test_area=test_area,
                                                      n_estimators=100)
        fi_df = fi_df.append(feature_imp, sort=True, ignore_index=True)

    # Store importance of attributes of each test_area
    fi_avg = {}
    fi_df = fi_df.drop(pred_var, axis=1)
    for col in fi_df.columns:
        fi_avg[col] = fi_df[col].mean()

    # Extract average of each attribute
    if isinstance(comparison_df, pd.DataFrame):
        s = pd.Series(fi_avg, name=pred_var[0])
        comparison_df = comparison_df.append(s)
    else:
        s = pd.Series(fi_avg, name=pred_var[0])
        comparison_df = pd.DataFrame(None, columns=fi_df.columns)
        comparison_df = comparison_df.append(s)

# Plot i
result = comparison_df.transpose()
result.plot(y=['norm', 'fs'], kind="bar")
plt.xlabel('Attribute')
plt.ylabel('Attributes Importance')



# ----------------------------------------------------------------------------------------------
# PLOT ii: removing most important variable
# ----------------------------------------------------------------------------------------------
most_imp = []
pred_var = ('norm', '1.0')
# df_attr = pd.read_csv(str(study_area_dir) + '/attribute_table_AVG_T.csv', sep=",", index_col='study_area')
df_attr = pd.read_csv('C:/Users/Ion/TFM/data/study_areas/attribute_table_AVG_T.csv', sep=",", index_col='study_area')
df_attr = pd.get_dummies(df_attr)
area_pred_list = []
comparison_df = None
for i in range(len(df_attr.columns)):
    df, check_df = prepare_df(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
                              sim_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/two_points',
                              out_path='C:/Users/Ion/TFM/data/plots/regression/randomForest',
                              pred_var=pred_var,
                              drop_attr=most_imp,
                              select_attr=None)
    # print(df.columns)

    # Iterate over areas to predict on them
    fi_df = pd.DataFrame(None, columns=df.drop(pred_var, axis=1).columns)
    errors = []
    for test_area in df.index:
        error_nfs, feature_imp, pred_nfs, real_nfs = randomForest(df=df,
                                                                  test_area=test_area,
                                                                  n_estimators=5000)

        fi_df = fi_df.append(feature_imp, sort=True, ignore_index=True)
        # real_fs = check_df.loc[test_area, ('fs', '1.0')]
        real_fs = list(check_df.loc[test_area, ('fs', '0.05'): ('fs', '1.0')])[-1]
        pred_fs = check_df.loc[test_area, 'CarPt_users'] * pred_nfs
        error_fs = abs(1-(pred_fs/real_fs))
        errors.append(error_nfs)

        # append areas prediction to df
        area_pred_list.append([test_area, pred_nfs, float(real_nfs), error_nfs, pred_fs, real_fs, error_fs])

    # Store importance of attributes of each test_area
    # fi_df = fi_df.drop(pred_var, axis=1)
    fi_avg = {}
    for col in fi_df.columns:
        fi_avg[col] = fi_df[col].mean()

    # Select max attribute and add to list to exclude it in following iteration
    max_attr = list({k: v for k, v in sorted(fi_avg.items(), reverse=True, key=lambda item: item[1])})[0]
    most_imp.append(max_attr)

    fi_avg['importance'] = fi_avg[max_attr]
    fi_avg['avg_error'] = sum(errors) / len(errors)
    print(len(most_imp), max_attr, fi_avg[max_attr])

    # Extract average of each attribute
    if isinstance(comparison_df, pd.DataFrame):
        s = pd.Series(fi_avg, name=max_attr)
        comparison_df = comparison_df.append(s)
    else:
        s = pd.Series(fi_avg, name=max_attr)
        comparison_df = pd.DataFrame(None, columns=fi_df.columns)
        comparison_df = comparison_df.append(s)

    # Add most important attribute

# for i in range(1, len(area_pred_list),15):
#     print(area_pred_list[i])

# Get best predictino of each study_area-
area_pred_df = pd.DataFrame(area_pred_list, columns=['study_area', 'pred_nfs', 'real_nfs', 'error_nfs', 'pred_fs', 'real_fs', 'error_fs'])
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
pred_data_list = []
for area in set(area_pred_df['study_area']):
    area_df = area_pred_df[area_pred_df['study_area'] == area].reset_index()
    best_pred = min(area_df['error_nfs'])
    best_pred_ix = area_df['error_nfs'].idxmin()
    attribute = list(comparison_df.index)[best_pred_ix]
    pred_data_list.append([area, area_type_dict[area], best_pred, best_pred_ix, attribute])

pred_data_df = pd.DataFrame(pred_data_list, columns=['study_area', 'area_type', 'best_pred', 'best_pred_ix', 'attribute'])
pred_data_df = pred_data_df.set_index('study_area')


# Plot ii
# result = comparison_df.transpose()
# comparison_df.plot(y=['importance'], kind="bar")
# comparison_df.plot.line(y=['avg_error'], color='r')
# plt.xlabel('Attribute')
# plt.ylabel('Attributes Importance')


fig, ax = plt.subplots()

ax.bar(comparison_df.index, comparison_df['importance'], label='Attribute importance')
ax.set_ylabel("Attribute importance / Prediction Avg Error (%)", fontsize=12)

ax.plot(comparison_df.index, comparison_df['avg_error'], color='r', label='Average Error')

plt.xticks(rotation=90)

ax2 = ax.twinx()
ax2.invert_yaxis()
bar_data_list=[list(comparison_df.index)]
colors = []
for area in set(area_pred_df['study_area']):
    area_df = pred_data_df.loc[area]
    bar_data = []
    area_type = area_df.loc['area_type']
    ix = area_df.loc['best_pred_ix']
    for i in range(len(comparison_df)):
        ix = area_df.loc['best_pred_ix']
        if i == ix:
            bar_data.append(area_df.loc['best_pred'])
        else:
            bar_data.append(0)
    if area_type == 'rural':
        color = 'm'
    elif area_type == 'urban':
        color = 'b'
    elif area_type == 'mountain':
        color = 'g'

    bar_data_list.append(bar_data)
    colors.append(color)
for x, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15 in zip(bar_data_list[0], bar_data_list[1], bar_data_list[2], bar_data_list[3], bar_data_list[4], bar_data_list[5], bar_data_list[6], bar_data_list[7], bar_data_list[8], bar_data_list[9], bar_data_list[10], bar_data_list[11], bar_data_list[12], bar_data_list[13], bar_data_list[14], bar_data_list[15]):
    for i, (h, c) in enumerate(sorted(zip([h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15], colors))):
        if h < 1:
            ax2.bar(x, h, color=c, zorder=-i)
            ax2.hlines(y=h, xmin=list(comparison_df.index).index(x), xmax=43, colors=c, linewidth=0.8, linestyles='dashdot', zorder=100)
top, bottom = ax2.get_ylim()
ax2.set_ylim(top*1.2, bottom)
ax2.set_ylabel("Best prediction by area type (% error)", fontsize=12)

# Add legend
urban_patch = mpatches.Patch(color='m', label='Urban Areas')
rural_patch = mpatches.Patch(color='b', label='Rural Areas')
mountain_patch = mpatches.Patch(color='g', label='Mountain Areas')
mountain_patch = mpatches.Patch(color='g', label='Mountain Areas')
mountain_patch = mpatches.Patch(color='g', label='Mountain Areas')
fig.legend(loc='best', ncol=1, prop={'size': 12}, frameon=True, handles=[urban_patch, rural_patch, mountain_patch])

plt.tight_layout()
plt.title('Random Forest prediction by removing most important attribute', fontsize=12, fontweight=0)



# ----------------------------------------------------------------------------------------------
# df = pd.get_dummies(df)
# print('The shape of our features is:', df.shape)
# df_detail = df.describe()
# print(pred_var, 'all attributes prediction')


# error_list = []
# for area in df.index:
#     error = randomForest(df, area)
#     error_list.append(error)
# print('error avg:', sum(error_list)/len(error_list))

