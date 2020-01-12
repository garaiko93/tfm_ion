import networkx as nx
import igraph as ig
print(ig.__file__)
import datetime
import pickle
from shapely.geometry import Point

print(datetime.datetime.now().time())

# graph_file = r'C:\Users\Ion\TFM\data\network_graphs'
# study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
# area = 'luzern'
# new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
# new_diG = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_DiGraph_largest.gpickle')
# shp_path = str(study_area_dir) + '/' + area
# file = open(str(graph_file) + "/ch_nodes_dict2056.pkl", 'rb')
# nodes_dict = pickle.load(file)
#

# # create igraph with networkx graph info
g = ig.Graph(directed=True)
# for node in new_G.nodes():
#     g.add_vertex(name=str(node))
# for edge in new_G.edges.data():
#     g.add_edge(str(edge[0]), str(edge[1]),
#                time=edge[2]['time'],
#                way_id=edge[2]['way_id'],
#                modes=edge[2]['modes'],
#                length=edge[2]['length'])
# print(len(g.vs),
#       len(g.es))

# def straight(nodes_dict, g, nodes_to_it=g.vs['name']):
#     node_straightness = {}
#     # node_dist = {}
#     n = 0
#     # i = '429340218'
#     # j = '429340228'
#     i = '307862771'
#     j = '1098133710'
#     for i in nodes_to_it:
#         print(n)
#         i_lonlat = Point(nodes_dict[str(i)])
#         dist_comp_list = []
#         m = 0
#         for j in g.vs['name']:
#             m += 1
#             # print('\r', m, end='')
#             # try:
#             #     dist_comp = node_dist[(j, i)]
#             # except:
#             if i == j:
#                 continue
#             sp_dist = 0
#             j_lonlat = Point(nodes_dict[str(j)])
#             eucl_dist = i_lonlat.distance(j_lonlat)
#             path = g.get_shortest_paths(v=i, to=j, weights='time', mode='OUT', output="epath")
#             for k in path[0]:
#                 sp_dist += g.es[k]['length']
#                 # print(g.es[k]['length'])
#             dist_comp = eucl_dist / sp_dist
#             # node_dist[(i, j)] = dist_comp
#             dist_comp_list.append(dist_comp)
#         straightness = (1 / (len(g.vs)-1)) * sum(dist_comp_list)
#         node_straightness[i] = straightness
#         n += 1
#     return node_straightness
#
# def dict_data(dicti, shp_path, attrib_name):
#     min_val = min(dicti.values())
#     max_val = max(dicti.values())
#     avg_val = sum(dicti.values()) / len(dicti)
#     len_val = len(dicti)
#     # with open(str(shp_path) + '/attr_' + str(attrib_name) + '.pkl', 'wb') as f:
#     #     pickle.dump(dicti, f, pickle.HIGHEST_PROTOCOL)
#     print(str(attrib_name) + ' (len,min,max,avg): ' +
#           str(len_val) + ', ' + str(min_val) + ', ' + str(max_val) + ', ' + str(avg_val))
#     return str([len_val, min_val,max_val,avg_val])

itime = datetime.datetime.now().time()
# node_straightness = straight(nodes_dict, g)
# attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
# node_straightness_data = dict_data(node_straightness, shp_path, 'node_straightness')
# attr_df.at['node_straightness', area] = node_straightness_data
# attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
print(itime)
print(datetime.datetime.now().time())





# g.ecount()
# g.es['modes']
# g.es[3152].source
# g.vs[1415]['name']
# g.es['way_id'] == 999969
# print(g)
# g.shortest_paths_dijkstra(source='429340270', target='2677593027', weights='time', mode='ALL')
# path = g.get_shortest_paths(v='429340270', to='2677593027', weights='time', mode='OUT', output="epath")
# for i in path[0]:
#     print(g.es[i]['length'])
# g.vs['name']
# g.es[0]