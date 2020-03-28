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
import datetime


parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
parser.add_argument('--area-path', dest="area_path", help='path to simulation_output folder')
parser.add_argument('--area', dest="area", help='area of simulation')
parser.add_argument('--path', dest="path", help='path to simulation_output folder')
parser.add_argument('--fleet-size', dest="fleet_size", help='fleet-size of simulation')
args = parser.parse_args()


area_path = args.path
area = args.area
av_share = args.av_share
fleet_size = args.fleet_size

# sim_path = str(area_path) + '/simulations/' + str(av_share) + '/' + str(fleet_size) + '/simulation_output'
sim_path = r'C:/Users/Ion/TFM/data/study_areas/' + str(area) + '/simulations/' + str(av_share) + '/' + str(fleet_size) + \
       '/simulation_output'

df = pd.read_csv(str(sim_path) + '/av_passenger_rides.csv', sep=";")
waiting_times = [x for x in list(df['waiting_time']) if str(x) != 'nan']
avg_wt = sum(waiting_times)/len(waiting_times)
var_wt = np.var(waiting_times)
stdev_wt = statistics.stdev(waiting_times)

# print(len(df), len(waiting_times), avg_wt)

# Check if .csv with attributes exists:
if os.path.isfile(str(area_path) + '/simulations/simulations_df.csv'):
    avg_df = pd.read_csv(str(area_path) + '/simulations/avg_df.csv', sep=",", index_col='attributes', dtype=object)
    var_df = pd.read_csv(str(area_path) + '/simulations/var_df.csv', sep=",", index_col='attributes',
                         dtype=object)
    stdev_df = pd.read_csv(str(area_path) + '/simulations/stdev_df.csv', sep=",", index_col='attributes',
                         dtype=object)
    print(datetime.datetime.now(), 'Simulation df already exist, file loaded')
else:
    avg_df = pd.DataFrame(data=None)
    var_df = pd.DataFrame(data=None)
    stdev_df = pd.DataFrame(data=None)
    print(datetime.datetime.now(), 'Simulation df do not exist, avg_df var_df and stdev_df were created empty.')









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
