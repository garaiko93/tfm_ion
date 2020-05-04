import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def data_setup(study_area_dir, sim_path, result_path, regression_func, predict_on):
    df_lr = pd.read_csv(str(sim_path) + '/linear_regression.csv', sep=",", index_col='area')

    df_sim_N = pd.read_csv(str(sim_path) + '/sim_opt_fs_norm.csv', sep=",", index_col='area')
    df_sim = pd.read_csv(str(sim_path) + '/sim_opt_fs.csv', sep=",", index_col='area')
    df_sim_N.columns = pd.MultiIndex.from_tuples([('norm', c) for c in df_sim_N.columns])
    df_sim.columns = pd.MultiIndex.from_tuples([('fs', c) for c in df_sim.columns])

    df = pd.read_csv(str(study_area_dir) + '/attribute_table_AVG_T.csv', sep=",", index_col='study_area')

    full_df = pd.concat([df, df_sim_N, df_sim, df_lr], axis=1, sort=False)
    full_df = full_df.sort_values(by=['network_distance'], ascending=False)

    for predic in predict_on:
        print(predic)
    for test_area in full_df.index:
        # apply_regression(full_df, regression_func, predict_on, area, result_path)
        # call regression method as many things as necessary
        params = []
        test_data = []
        for predict in predict_on:
            x, y, test_elem = setup_xy(full_df, predict, test_area)
            param = regression_func(x, y, predict)

            test_data.append(test_elem)
            params.append(param)

        # test regression
        results = test_func(test_data, predict_on, params)

        # store results of prediction
        store_results(results, result_path)

    df = pd.read_csv(str(result_path) + '/regression_ab.csv', sep=",", index_col='study_area')

    print(sum(df['%_fs'])/len(df))


def setup_xy(full_df, predict_on, test_areas=None):
    predict_var = predict_on[0]
    predict_attr = predict_on[1]
    test = predict_on[2]

    if predict_attr is None:
        predict_attr = list(full_df.columns)
        drop_col = ['area_type',                ('norm', '0.05'),
                    ('norm', '0.1'),              ('norm', '0.2'),
                    ('norm', '0.4'),              ('norm', '0.6'),
                    ('norm', '0.8'),              ('norm', '1.0'),
                     ('fs', '0.05'),                ('fs', '0.1'),
                      ('fs', '0.2'),                ('fs', '0.4'),
                      ('fs', '0.6'),                ('fs', '0.8'),
                      ('fs', '1.0'), 'a', 'b']
        for item in drop_col:
            predict_attr.remove(item)

    if test_areas is None:
        test_areas = [random.choice(full_df.index)]
        # test_areas = []
        # for area_type in ['urban', 'rural', 'mountain']:
        #     random_area = random.choice(full_df[full_df['area_type'] == area_type].index)
        #     test_areas.append(random_area)

    full_train = full_df.drop(test_areas)
    full_test = full_df.loc[test_areas, :]

    # train_X = full_train[predict_attr].as_matrix()
    train_X = full_train[predict_attr].values
    # train_y = np.array(full_train[predict_var])
    train_y = full_train[predict_var].values

    test_X = full_test[predict_attr].values
    test_y = np.array(full_test[predict_var])
    test_carpt_users = np.array(full_test['CarPt_users'])
    test_fs = full_test[('fs', '1.0')]
    test_nfs = full_test[('norm', '1.0')]

    test_data = [test_X, test_y, test_carpt_users, test_fs, test_nfs, test_areas]

    return train_X, train_y, test_data


# -----------------------------------------------------------------------------
# OLS: ORDINARY LEAST SQUARES
# -----------------------------------------------------------------------------
def OLS_regression(X, y, predict_on):
    var_name = predict_on[1]

    model = sm.OLS(y, X)
    results = model.fit()

    # print(results.summary(xname=var_name))
    # print('Parameters: ', results.params)
    # print('R2: ', results.rsquared)

    return results.params

# -----------------------------------------------------------------------------
# OLS: ORDINARY LEAST SQUARES
# -----------------------------------------------------------------------------
def another_regression(X, y, predict_on):
    var_name = predict_on[1]

    model = sm.OLS(y, X)
    results = model.fit()

    # print(results.summary(xname=var_name))
    # print('Parameters: ', results.params)
    # print('R2: ', results.rsquared)

    return results.params


def test_func(test_data, predict_on, params):
    carpt_users = test_data[0][2]
    real_fs = test_data[0][3]
    real_nfs = test_data[0][4]
    test_areas = test_data[0][5]

    # values from regression on a parameter
    a_test_X = test_data[0][0]
    a_params = params[0]
    pred_a = np.dot(a_test_X, a_params)
    real_a = test_data[0][1]

    # values from regression on b parameter
    b_test_X = test_data[1][0]
    b_params = params[1]
    pred_b = np.dot(b_test_X, b_params)
    real_b = test_data[1][1]

    pred_nfs = (pred_a*100) + pred_b
    pred_fs = np.rint(pred_nfs * carpt_users)

    real_values = [real_fs, real_a, real_b, real_nfs]
    pred_values = [pred_fs, pred_a, pred_b, pred_nfs]

    # error = abs(1 - (pred_fs / real_fs))
    # error = pred_fs / real_fs

    # print('test areas:', test_areas)
    # print('pred_fs:', pred_fs)
    # print('real_fs:', real_fs)
    # print('error: ', error)

    return [test_areas, pred_values, real_values]

def store_results(results, result_path):
    if os.path.isfile(str(result_path) + '/regression_ab.csv'):
        df = pd.read_csv(str(result_path) + '/regression_ab.csv', sep=",", index_col='study_area')
    else:
        df = pd.DataFrame(data=None, columns=['real_a', 'pred_a', '%_a', 'real_b', 'pred_b', '%_b', 'real_nfs', 'pred_nfs', '%_nfs', 'real_fs', 'pred_fs', '%_fs'])

    # define new row
    area = results[0]
    pred_fs = np.array(results[1][0])
    pred_a = np.array(results[1][1])
    pred_b = np.array(results[1][2])
    pred_nfs = np.array(results[1][3])
    # pred_b = np.array(0)

    real_fs = np.array(results[2][0])
    real_a = np.array(results[2][1])
    real_b = np.array(results[2][2])
    real_nfs = np.array(results[2][3])

    # rmse = sqrt(mean_squared_error(real_fs, pred_fs))
    # error = abs(1 - (pred_fs / real_fs))
    error_a = pred_a / real_a
    error_b = pred_b / real_b
    # error_b = 0
    # error_fs = "{:.3f}".format(pred_fs / real_fs)
    error_fs = "{:.3f}".format(abs(1 - (pred_fs / real_fs)))
    error_nfs = pred_nfs / real_nfs

    # append new row to df
    df.loc[area] = [real_a, pred_a, error_a, real_b, pred_b, error_b, real_nfs, pred_nfs, error_nfs, real_fs, pred_fs, error_fs]
    # save df
    df.to_csv(str(result_path) + '/regression_ab.csv', sep=",", index=True, index_label='study_area')

predict_on = [
    ['a', ['population_density', 'btw_acc_trip_generation', 'btw_acc_trip_production', 'degree_centrality'], True],
    # ['a', ['population', 'CarPt_users', 'intersection_d_km', 'edge_load_centrality'], True],
    # ['a', ['CarPt_users', 'population_gini', 'avg_shortest_path_duration', 'efficiency', 'diameter'], True],
    # ['a', ['CarPt_users', 'population'], True],
    # ['b', ['network_distance', 'avg_degree', 'avg_neighbor_degree', 'avg_degree_connectivity', 'btw_acc_trip_generation', 'btw_acc_trip_production'], True]
    # ['b', ['network_distance', 'avg_degree', 'avg_neighbor_degree', 'avg_degree_connectivity', 'btw_acc_trip_generation', 'btw_acc_trip_production'], True]
    # ['b', ['network_distance', 'avg_degree', 'CarPt_users', 'avg_degree', 'btw_acc_trip_generation'], True]
    ['b', ['efficiency', 'street_d_km', 'population_density', 'diameter'], True]
    # ['b', ['avg_link_time', 'circuity_avg', 'degree_centrality', 'edge_betweenness'], True]
            ]

# predict_on = [[('norm', '1.0'), ['n_intersection', 'CarPt_users', 'edge_betweenness'], True]]

data_setup(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
           sim_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/two_points',
           result_path='C:/Users/Ion/TFM/data/plots/regression/ols',
           regression_func=OLS_regression,
           predict_on=predict_on)


# def data_setup(study_area_dir, sim_path, result_path, regression_func, predict_on):
#     df_lr = pd.read_csv(str(sim_path) + '/' + 'linear_regression.csv', sep=",", index_col='area')
#
#     df_sim_N = pd.read_csv(str(sim_path) + '/' + 'sim_opt_fs_norm.csv', sep=",", index_col='area')
#     df_sim = pd.read_csv(str(sim_path) + '/' + 'sim_opt_fs.csv', sep=",", index_col='area')
#     df_sim_N.columns = pd.MultiIndex.from_tuples([('norm', c) for c in df_sim_N.columns])
#     df_sim.columns = pd.MultiIndex.from_tuples([('fs', c) for c in df_sim.columns])
#
#     df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')
#
#     full_df = pd.concat([df, df_sim_N, df_sim, df_lr], axis=1, sort=False)
#     full_df = full_df.sort_values(by=['network_distance'], ascending=False)
#
#     for test_area in full_df.index:
#         # apply_regression(full_df, regression_func, predict_on, area, result_path)
#
#         # call regression method as many things as necessary
#         if len(predict_on) == 1:
#             # predict_on = predict_on[0]
#             x, y, test_data = setup_xy(full_df, predict_on[0], test_area)
#             params = regression_func(x, y, predict_on[0])
#             pred = predict_on[0][0]
#         else:
#             params = []
#             test_data = []
#             pred = 'ab'
#             # test_areas = None
#             for predict in predict_on:
#                 x, y, test_elem = setup_xy(full_df, predict, test_area)
#                 param = regression_func(x, y, predict)
#
#                 # test_areas = test_data[4]
#
#                 test_data.append(test_elem)
#                 params.append(param)
#
#         # test regression
#         results = test_func(test_data, predict_on, params)
#
#         # store results of prediction
#         store_results(results, result_path, pred)
#
#
# def setup_xy(full_df, predict_on, test_areas=None):
#     predict_var = predict_on[0]
#     predict_attr = predict_on[1]
#     test = predict_on[2]
#
#     if len(predict_attr) == 0:
#         predict_attr = list(full_df.columns)
#         drop_col = ['area_type',                ('norm', '0.05'),
#                     ('norm', '0.1'),              ('norm', '0.2'),
#                     ('norm', '0.4'),              ('norm', '0.6'),
#                     ('norm', '0.8'),              ('norm', '1.0'),
#                      ('fs', '0.05'),                ('fs', '0.1'),
#                       ('fs', '0.2'),                ('fs', '0.4'),
#                       ('fs', '0.6'),                ('fs', '0.8'),
#                       ('fs', '1.0'), 'a', 'b']
#         for item in drop_col:
#             predict_attr.remove(item)
#
#     if test:
#         if test_areas is None:
#             test_areas = []
#             # for area_type in ['urban', 'rural', 'mountain']:
#             #     random_area = random.choice(full_df[full_df['area_type'] == area_type].index)
#             #     test_areas.append(random_area)
#             test_areas = [random.choice(full_df.index)]
#
#         full_train = full_df.drop(test_areas)
#         full_test = full_df.loc[test_areas, :]
#
#         train_X = full_train[predict_attr].as_matrix()
#         train_y = np.array(full_train[predict_var])
#
#         test_X = full_test[predict_attr].as_matrix()
#         test_y = np.array(full_test[predict_var])
#         test_carpt_users = np.array(full_test['CarPt_users'])
#         test_fs = full_test[('fs', '1.0')]
#
#         test_data = [test_X, test_y, test_carpt_users, test_fs, test_areas]
#     else:
#         train_X = full_df[predict_attr].as_matrix()
#         train_y = np.array(full_df[predict_var])
#         test_data = None
#
#     return train_X, train_y, test_data
#
#
# # -----------------------------------------------------------------------------
# # OLS: ORDINARY LEAST SQUARES
# # -----------------------------------------------------------------------------
# def OLS_regression(X, y, predict_on):
#     var_name = predict_on[1]
#
#     model = sm.OLS(y, X)
#     results = model.fit()
#
#     # print(results.summary(xname=var_name))
#     # print('Parameters: ', results.params)
#     # print('R2: ', results.rsquared)
#
#     return results.params
#
# # -----------------------------------------------------------------------------
# # OLS: ORDINARY LEAST SQUARES
# # -----------------------------------------------------------------------------
# def another_regression(X, y, predict_on):
#     var_name = predict_on[1]
#
#     model = sm.OLS(y, X)
#     results = model.fit()
#
#     # print(results.summary(xname=var_name))
#     # print('Parameters: ', results.params)
#     # print('R2: ', results.rsquared)
#
#     return results.params
#
#
# def test_func(test_data, predict_on, params):
#     if len(predict_on) == 1:
#         test_X = test_data[0]
#         carpt_users = test_data[2]
#         real_fs = test_data[3]
#         test_areas = test_data[4]
#
#         pred_y = np.dot(test_X, params)
#         pred_fs = pred_y * carpt_users
#
#         real_values = [real_fs, None, None]
#         # error = pred_fs / real_fs
#     else:
#         carpt_users = test_data[0][2]
#         real_fs = test_data[0][3]
#         test_areas = test_data[0][4]
#
#         a_test_X = test_data[0][0]
#         a_params = params[0]
#         pred_a = np.dot(a_test_X, a_params)
#         real_a = test_data[0][1]
#
#         b_test_X = test_data[1][0]
#         b_params = params[1]
#         pred_b = np.dot(b_test_X, b_params)
#         real_b = test_data[1][1]
#
#         pred_fs = ((pred_a*100) + pred_b) * carpt_users
#         real_values = [real_fs, real_a, real_b]
#
#         # error = abs(1 - (pred_fs / real_fs))
#         # error = pred_fs / real_fs
#
#     # print('test areas:', test_areas)
#     # print('pred_fs:', pred_fs)
#     # print('real_fs:', real_fs)
#     # print('error: ', error)
#
#     return [test_areas, pred_fs, real_values]
#
# def store_results(results, result_path, pred):
#     if os.path.isfile(str(result_path) + '/regression_' + str(pred) + '.csv'):
#         df = pd.read_csv(str(result_path) + '/regression_' + str(pred) + '.csv', sep=",", index_col='study_area')
#     else:
#         df = pd.DataFrame(data=None, columns=['pred_fs', 'real_fs', 'error'])
#
#     # define new row
#     area = results[0]
#     pred_fs = np.array(results[1])
#
#     real_fs = np.array(results[2][0])
#     real_a = np.array(results[2][1])
#     real_b = np.array(results[2][2])
#     # rmse = sqrt(mean_squared_error(real_fs, pred_fs))
#     # error = abs(1 - (pred_fs / real_fs))
#     error = pred_fs / real_fs
#
#     if real_a is None:
#         # append new row to df
#         df.loc[area] = [real_fs, pred_fs, error]
#         # save df
#         df.to_csv(str(result_path) + '/regression_' + str(pred) + '.csv', sep=",", index=True, index_label='study_area')
#     else:
#         # append new row to df
#         df.loc[area] = [real_fs, pred_fs, error]
#         # save df
#         df.to_csv(str(result_path) + '/regression_' + str(pred) + '.csv', sep=",", index=True, index_label='study_area')
# predict_on = [
#     ['a', ['CarPt_users', 'population', 'edge_betweenness', 'btw_acc_trip_generation', 'btw_acc_trip_production', 'degree_centrality'], True],
#     ['b', ['network_distance', 'avg_degree', 'avg_neighbor_degree', 'avg_degree_connectivity', 'btw_acc_trip_generation', 'btw_acc_trip_production'], True]
#             ]
# # predict_on = [[('norm', '1.0'), ['n_intersection', 'CarPt_users', 'edge_betweenness'], True]]
#
# data_setup(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
#            sim_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/two_points',
#            result_path='C:/Users/Ion/TFM/data/plots/regression/ols',
#            regression_func=OLS_regression,
#            predict_on=predict_on)