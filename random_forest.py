import pandas as pd
import numpy as np
import os
import datetime

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt


study_area_dir='C:/Users/Ion/TFM/data/study_areas'
sim_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/two_points'
result_path='C:/Users/Ion/TFM/data/plots/regression/ols/try8'


# def data_setup(study_area_dir, sim_path, result_path, regression_func, predict_on):
# Check if out_path exists or create it
if not os.path.exists(str(result_path)):
    os.makedirs(str(result_path))
    print('Directory created')
else:
    print('Directory exists, change name of attempt output directory')

df_lr = pd.read_csv(str(sim_path) + '/linear_regression.csv', sep=",", index_col='area')

df_sim_N = pd.read_csv(str(sim_path) + '/sim_opt_fs_norm.csv', sep=",", index_col='area')
df_sim = pd.read_csv(str(sim_path) + '/sim_opt_fs.csv', sep=",", index_col='area')
df_sim_N.columns = pd.MultiIndex.from_tuples([('norm', c) for c in df_sim_N.columns])
df_sim.columns = pd.MultiIndex.from_tuples([('fs', c) for c in df_sim.columns])

df_attr = pd.read_csv(str(study_area_dir) + '/attribute_table_AVG_T.csv', sep=",", index_col='study_area')

full_df = pd.concat([df_attr, df_sim_N, df_sim, df_lr], axis=1, sort=False)
full_df = full_df.sort_values(by=['network_distance'], ascending=False)


# Concatenate dfs with feature to predict
pred_var = ('norm', '1.0')
# pred_var = ('fs', '1.0')
# attr_sel = ['population_density', 'CarPt_users', 'efficiency', 'btw_acc_trip_generation', 'node_straightness']
attr_sel = ['CarPt_users', 'population_density', 'area']
# df = pd.concat([df_attr[attr_sel], df_sim_N[pred_var]], axis=1, sort=False)
# df = pd.concat([df_attr[attr_sel], df_sim[pred_var]], axis=1, sort=False)

# df = pd.concat([df_attr, df_sim[pred_var]], axis=1, sort=False)
df = pd.concat([df_attr, df_sim_N[pred_var]], axis=1, sort=False)
# df = pd.concat([df_attr.drop(['node_load_centrality', 'edge_betweenness'], axis=1), df_sim_N[pred_var]], axis=1, sort=False)

# Replace area_type ('mountain', 'urban', 'rural') for numerical features
df = pd.get_dummies(df)
# print('The shape of our features is:', df.shape)
# df_detail = df.describe()
for area in df.index:
    randomForest(df, area)

def randomForest(df, test_area):
    # test_area = 'lausanne'
    print(test_area)
    full_train = df.drop(test_area)
    full_test = df.loc[test_area, :]
    feature_list = list(df.drop(pred_var, axis=1).columns)

    train_features = np.array(full_train.drop(pred_var, axis=1))
    train_labels = np.array(full_train[pred_var])

    test_features = np.array(full_test.drop(pred_var)).reshape(1, -1)
    test_labels = np.array(full_test[pred_var])

    # # Features and Targets and Convert Data to Arrays
    # # Labels are the values we want to predict
    # labels = np.array(df[pred_var])
    # # Remove the labels from the features
    # # axis 1 refers to the columns
    # df = df.drop(pred_var, axis=1)
    # # Saving feature names for later use
    # feature_list = list(df.columns)
    # # Convert to numpy array
    # features = np.array(df)
    # # Split the data into training and testing sets
    # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=(1/len(features)), random_state=42)
    # # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)


    # print('Training Features Shape:', train_features.shape)
    # print('Training Labels Shape:', train_labels.shape)
    # print('Testing Features Shape:', test_features.shape)
    # print('Testing Labels Shape:', test_labels.shape)

    # The baseline predictions are the historical averages: 20% error
    # baseline_preds = 0.8 * test_labels
    # baseline_errors = abs(1 - (baseline_preds / test_labels))
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=100)
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # path = rf.decision_path(test_features)
    errors = abs(1 - (predictions / test_labels))
    print(test_area, 'Error:', round(np.mean(errors), 4), predictions, test_labels)



    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    i = 0
    if len(feature_importances) < 10:
        limit = len(feature_importances)
    else:
        limit = 10
    while i < limit:
        print('Variable: ', feature_importances[i][0], 'Importance: ', feature_importances[i][1])
        i +=1

    # list of x locations for plotting
    # x_values = list(range(len(importances)))
    # # Make a bar chart
    # plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
    # # Tick labels for x axis
    # plt.xticks(x_values, feature_list, rotation='vertical')
    # # Axis labels and title
    # plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


out_file = 'C:/Users/Ion/TFM/data/plots/regression/randomForest'
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file=out_file + '/tree.dot', feature_names=feature_list, rounded=True, precision=4)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file(out_file + '/tree.dot')
# Write graph to a png file
graph.write_png(out_file + '/tree.png')
