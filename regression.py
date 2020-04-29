import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def regression_data_setup(study_area_dir, sim_path, reg_func, predict_on):
    sim_path = 'C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/' + str(fit_func)
    study_area_dir = 'C:/Users/Ion/TFM/data/study_areas'

    df_lr = pd.read_csv(str(sim_path) + '/' + 'linear_regression.csv', sep=",", index_col='area')

    df_sim_N = pd.read_csv(str(sim_path) + '/' + 'sim_opt_fs_norm.csv', sep=",", index_col='area')
    df_sim = pd.read_csv(str(sim_path) + '/' + 'sim_opt_fs.csv', sep=",", index_col='area')
    df_sim_N.columns = pd.MultiIndex.from_tuples([('norm', c) for c in df_sim_N.columns])
    df_sim.columns = pd.MultiIndex.from_tuples([('fs', c) for c in df_sim.columns])

    df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table_AVG_T.csv', sep=",", index_col='study_area')

    full_df = pd.concat([df, df_sim_N, df_sim, df_lr], axis=1, sort=False)
    full_df = full_df.sort_values(by=['network_distance'], ascending=False)

    x, y, test_data = setup_xy(full_df, predict_on)
    reg_func(x, y, predict_on, test_data)


def setup_xy(full_df, predict_on):
    predict_var = predict_on[0]
    predict_attr = predict_on[1]
    test = predict_on[2]

    if len(predict_attr) == 0:
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

    if test:
        test_areas = []
        for area_type in ['urban', 'rural', 'mountain']:
            random_area = random.choice(full_df[full_df['area_type'] == area_type].index)
            test_areas.append(random_area)

        full_train = full_df.drop(test_areas)
        full_test = full_df.loc[test_areas, :]

        train_X = full_train[predict_attr].as_matrix()
        train_y = np.array(full_train[predict_var])

        test_X = full_test[predict_attr].as_matrix()
        test_y = np.array(full_test[predict_var])
        test_carpt_users = np.array(full_test['CarPt_users'])
        test_fs = np.array(full_test[('fs', '1.0')])

        test_data = [test_X, test_y, test_carpt_users, test_fs, test_areas]
    else:
        train_X = full_df[predict_attr].as_matrix()
        train_y = np.array(full_df[predict_var])
        test_data = None

    return train_X, train_y, test_data

# -----------------------------------------------------------------------------
# OLS: ORDINARY LEAST SQUARES
# -----------------------------------------------------------------------------
def OLS_regression(X, y, predict_on, test_data):
    var_name = predict_on[1]

    model = sm.OLS(y, X)
    results = model.fit()

    # print(results.summary(xname=var_name))
    # print('Parameters: ', results.params)
    # print('R2: ', results.rsquared)

    # Test results
    if test_data:
        test_X = test_data[0]
        carpt_users = test_data[2]
        real_fs = test_data[3]
        test_areas = test_data[4]

        pred_y = np.dot(test_X, results.params)
        pred_fs = pred_y * carpt_users

        error = pred_fs / real_fs

        print('test areas:', test_areas)
        print('pred_fs:', pred_fs)
        print('rea_fs:', real_fs)
        print('error: ', error)








fit_func = 'two_points'

# test_areas = ['sion', 'plateau', 'lausanne']
predict_on = [('norm', '1.0'), ['n_intersection', 'CarPt_users', 'edge_betweenness', 'btw_acc_trip_generation'], True]
# predict_on = ['a', [''], []]
# predict_on = ['b', [''], []]


regression_data_setup(study_area_dir='C:/Users/Ion/TFM/data/study_areas',
                      sim_path='C:/Users/Ion/TFM/data/plots/sim_plots/wt_fs/' + str(fit_func),
                      reg_func=OLS_regression,
                      predict_on=predict_on)