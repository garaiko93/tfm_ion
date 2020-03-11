import pandas as pd
import pickle
import csv
import networkx as nx
from graph_analysis import straightness, dict_data


study_area_dir = '/cluster/home/gaion/TFM/study_areas'
area = 'freiburg'
file = open('/cluster/home/gaion/TFM/network_graphs/ch_nodes_dict2056.pkl', 'rb')
area_path = str(study_area_dir) + "/" + str(area)
nodes_dict = pickle.load(file)

count = 0
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



# new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
# len(new_G.nodes())
# Out[6]: 22553
# nodes_list = new_G.nodes()
# split_list = np.array_split(nodes_list, 10)
#
#
# count = 0
# for slist in split_list:
#     with open(str(study_area_dir) + '/' + str(area) + "/node_list" + str(count) + ".csv", "w") as output:
#         writer = csv.writer(output, lineterminator='\n')
#         for val in list(slist):
#             writer.writerow([val])
#     count += 1