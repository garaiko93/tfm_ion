import pandas as pd
import pickle
import csv
import numpy as np
from os import walk
import argparse
import networkx as nx

from graph_analysis import straightness, dict_data, take_avg

parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
parser.add_argument('--count', dest="count", help='path to study areas')
# parser.add_argument('--network-graphs', dest="network_graphs", help="path to network_graphs")
args = parser.parse_args()
# filter_graph(args.study_areas, args.network_graphs)


study_area_dir = '/cluster/home/gaion/TFM/study_areas'
area = 'freiburg'
file = open('/cluster/home/gaion/TFM/network_graphs/ch_nodes_dict2056.pkl', 'rb')
# file = open(str(study_area_dir) + '/freiburg/node_straightness0.pkl', 'rb')
nodes_dict = pickle.load(file)

area_path = str(study_area_dir) + "/" + str(area)

count = args.count
# count = 0
print(count)
nodes_list = []
with open(str(study_area_dir) + "/" + str(area) + "/node_list" + str(count) + ".csv", 'r') as f:
    reader = csv.reader(f)
    for i in reader:
        nodes_list.append(i[0])

shp_path = str(study_area_dir) + "/" + area
new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)

# ----------------------------------------------------------------------
# 8. STRAIGHTNESS CENTRALITY: compares the shortest path with the euclidean distance of each pair of nodes
node_straightness = straightness(nodes_dict, new_G, nodes_list)

# node_straightness_data = dict_data(node_straightness, shp_path, 'node_straightness')
# dict_data(node_straightness, area_path, 'node_straightness', area, attr_df)
# attr_df.at['node_straightness', area] = node_straightness_data
# attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
with open(str(area_path) + '/node_straightness' + str(count) + '.pkl', 'wb') as f:
    pickle.dump(node_straightness, f, pickle.HIGHEST_PROTOCOL)
print('----------------------------------------------------------------------')
print('Process finished correctly: shp and graph files created in destination')

# ----------------------------------------------------------------------
# SPLIT LIST OF NODES INTO X EQUAL PARTS

# study_area_dir = r'C:/Users/Ion/TFM/data/study_areas'
# area = 'chur'
# new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
# print(len(new_G.nodes()))
# nodes_list = new_G.nodes()
# split_list = np.array_split(nodes_list, 20)
# print(len(split_list))
# print(len(split_list[0]))
#
# count = 0
# for slist in split_list:
#     with open(str(study_area_dir) + '/' + str(area) + "/node_list" + str(count) + ".csv", "w") as output:
#         writer = csv.writer(output, lineterminator='\n')
#         for val in list(slist):
#             writer.writerow([val])
#     count += 1

# ----------------------------------------------------------------------
# MERGE THE OUTPUT DICTIONARIES OF EACH SPLITTED LIST OF NODES

study_area_dir = r'C:/Users/Ion/TFM/data/study_areas'
area = 'chur'
area_path = str(study_area_dir) + '/' + area
f = []
for (dirpath, dirnames, filenames) in walk(area_path):
    f.extend(filenames)
    break

straightness_dict = {}
for file in f:
    if 'node_straightness' in file:
        file = open(str(area_path) + '/' + str(file), 'rb')
        nodes_dict = pickle.load(file)

        straightness_dict = {**straightness_dict, **nodes_dict}
print(len(straightness_dict))

attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
attr_df = dict_data(straightness_dict, area_path, 'node_straightness', area, attr_df)
take_avg(attr_df, study_area_dir, attributes=None)
