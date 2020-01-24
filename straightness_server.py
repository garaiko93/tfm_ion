import pandas as pd
import pickle
import networkx as nx
from graph_analysis import straightness, dict_data

study_area_dir = '/cluster/home/gaion/TFM/study_areas'
area = 'neuchatel'
file = open('/cluster/home/gaion/TFM/network_graphs/ch_nodes_dict2056.pkl', 'rb')
nodes_dict = pickle.load(file)
shp_path = str(study_area_dir) + "/" + area
new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)

# ----------------------------------------------------------------------
# 8. STRAIGHTNESS CENTRALITY: compares the shortest path with the euclidean distance of each pair of nodes
node_straightness = straightness(nodes_dict, new_G)
node_straightness_data = dict_data(node_straightness, shp_path, 'node_straightness')
attr_df.at['node_straightness', area] = node_straightness_data
attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])

print('----------------------------------------------------------------------')
print('Process finished correctly: shp and graph files created in destination')

