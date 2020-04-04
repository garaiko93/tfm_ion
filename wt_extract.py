import statistics
import argparse
import os
import glob
import shutil
import ntpath
import numpy as np
import pandas as pd

# area = 'test_area'
# graph_file = r'C:\Users\Ion\TFM\data\network_graphs'
# area_path = r'C:\Users\Ion\TFM\data\study_areas/locarno'
# av_share = '0.2'
# fleet_size = 40
# df_name='avg_df'

def file_remove(path, match):
    # file removal
    for CleanUp in glob.glob(str(path) + '/*.*'):
        if not CleanUp.endswith(match):
            os.remove(CleanUp)
    # folder removal
    listOfFile = os.listdir(path)
    for folder in listOfFile:
        fullPath = os.path.join(path, folder)
        if os.path.isdir(fullPath):
            shutil.rmtree(str(path) + '/' + str(folder))

def df_update(area_path, fleet_size, av_share, wt, df_name):
    if os.path.isfile(str(area_path) + '/simulations/' + str(df_name) + '.csv'):
        df = pd.read_csv(str(area_path) + '/simulations/' + str(df_name) + '.csv', sep=",", index_col='fleet_size')
    else:
        df = pd.DataFrame(data=None)

    try:
        # Save actual simulation value in df
        df.at[fleet_size, av_share] = wt
    except:
        # Update attribute table with new added attributes or study areas
        if fleet_size not in df.index:
            s = pd.Series(name=fleet_size)
            df = df.append(s)
        # create empty row with areas name to add attributes
        if av_share not in df.columns:
            df.insert(loc=len(df.columns), column=av_share, value=['' for i in range(df.shape[0])])
        # Save actual simulation value in df
        df.at[fleet_size, av_share] = wt

    # Finally save df back
    df.to_csv(str(area_path) + '/simulations/' + str(df_name) + '.csv', sep=",", index=True, index_label='fleet_size')

def wt_extract_fc(area_path, fleet_size, av_share):
    # Defining simulation file to extract waiting times
    sim_path = str(area_path) + '/simulations/' + str(av_share) + '/' + str(fleet_size) + '/simulation_output'
    fs_path = str(area_path) + '/simulations/' + str(av_share)
    sim_file = str(sim_path) + '/av_passenger_rides.csv'

    # Delete all remaining files and folders in simulation_output folder to save space
    file_remove(sim_path, 'av_passenger_rides.csv')

    # Computation of waiting time values
    df = pd.read_csv(sim_file, sep=";")
    waiting_times = [x for x in list(df['waiting_time']) if str(x) != 'nan']
    avg_wt = float("{:.3f}".format(sum(waiting_times)/len(waiting_times)))
    var_wt = float("{:.3f}".format(np.var(waiting_times)))
    stdev_wt = float("{:.3f}".format(statistics.stdev(waiting_times)))


    # Record waiting time values in dataframes:
    if (avg_wt < 800) and (avg_wt > 100):
        df_update(area_path, fleet_size, av_share, avg_wt, 'avg_df')
        df_update(area_path, fleet_size, av_share, var_wt, 'var_df')
        df_update(area_path, fleet_size, av_share, stdev_wt, 'stdev_df')
    # else:
    #     # folder removal
    #     listOfFile = os.listdir(fs_path)
    #     for folder in listOfFile:
    #         if folder == str(fleet_size):
    #             fullPath = os.path.join(fs_path, folder)
    #             if os.path.isdir(fullPath):
    #                 shutil.rmtree(str(fs_path) + '/' + str(folder))

    #this gives the wt value to the bash file
    # print(avg_wt)
    return avg_wt

# Parsing command line arguments
# parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
# parser.add_argument('--area-path', dest="area_path", help='path to simulation_output folder')
# parser.add_argument('--fleet-size', dest="fleet_size", help='fleet-size of simulation')
# parser.add_argument('--av-share', dest="av_share", help='av_share value of simulation')
# args = parser.parse_args()
#
# area_path = args.area_path.split('\r')[0]
# fleet_size = int(args.fleet_size.split('\r')[0])
# av_share = args.av_share.split('\r')[0]

# avg_wt = wt_extract_fc(area_path, fleet_size, av_share)



