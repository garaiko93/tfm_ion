import re
import gzip
import statistics
import time
import pandas as pd
import pickle
import csv
import numpy as np
import statistics
import argparse
import os
import ntpath


def df_update(area_path, fleet_size, av_share, wt, df_name):
    if os.path.isfile(str(area_path) + '/simulations/' + str(df_name) + '.csv'):
        df = pd.read_csv(str(area_path) + '/simulations/' + str(df_name) + '.csv', sep=",", index_col='fleet_size', dtype=object)
        df = pd.read_csv(r'C:/Users/Ion/TFM/data/study_areas/zurich_kreis/simulations/avg_df.csv', sep=",", index_col='fleet_size', dtype=object)
    else:
        df = pd.DataFrame(data=None)

    # Update attribute table with new added attributes or study areas
    if fleet_size not in df.index:
        s = pd.Series(name=fleet_size)
        df = df.append(s)
        print('row added')

    # create empty row with areas name to add attributes
    if av_share not in df.columns:
        df.insert(loc=len(df.columns), column=av_share, value=['' for i in range(df.shape[0])])

    # Save actual simulation value in df
    df.at[fleet_size, av_share] = wt

    # Finally save df back
    df.to_csv(str(area_path) + '/simulations/' + str(df_name) + '.csv', sep=",", index=True, index_label='fleet_size')

def del_r(variable):
    filename = variable.split('\r')[0]
    return filename

parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
parser.add_argument('--area-path', dest="area_path", help='path to simulation_output folder')
parser.add_argument('--fleet-size', dest="fleet_size", help='fleet-size of simulation')
parser.add_argument('--av-share', dest="av_share", help='av_share value of simulation')
args = parser.parse_args()

area_path = args.area_path.split('\r')[0]
fleet_size = args.fleet_size.split('\r')[0]
av_share = args.av_share.split('\r')[0]

sim_path = str(area_path) + '/simulations/' + str(av_share) + '/' + str(fleet_size) + '/simulation_output'
# sim_path = r'C:/Users/Ion/TFM/data/study_areas/' + str(area) + '/simulations/' + str(av_share) + '/' + str(fleet_size) + \
#        '/simulation_output'


df = pd.read_csv(str(sim_path) + '/av_passenger_rides.csv', sep=";")
waiting_times = [x for x in list(df['waiting_time']) if str(x) != 'nan']
avg_wt = sum(waiting_times)/len(waiting_times)
var_wt = np.var(waiting_times)
stdev_wt = statistics.stdev(waiting_times)

# Record waiting time values in dataframes:
df_update(area_path, fleet_size, av_share, avg_wt, 'avg_df')
df_update(area_path, fleet_size, av_share, var_wt, 'var_df')
df_update(area_path, fleet_size, av_share, stdev_wt, 'stdev_df')

#this gives the wt value to the bash file
print(avg_wt)




# plans_file = r'C:/Users/Ion/TFM/simulations/auckland_example2/simulation_output/ITERS/it.40/40.plans.xml.gz'
# wt_list = []
# with gzip.open(plans_file) as f:
#     for line in f:
#         if b'type="av"' in line:
#             m = re.search(rb'"waitingTime":(.+),', line)
#             wt = float(m.group(1))
#             wt_list.append(wt)
# print('Plans in AVs with waitingTime values: ' + str(len(wt_list)))
# print('Average of all waitingTime values: ' + str(time.strftime('%H:%M:%S', time.gmtime(statistics.mean(wt_list)))))
