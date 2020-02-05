import multiprocessing
import threading
import pandas as pd
import pickle
import networkx as nx
import igraph
import pyproj
from shapely.geometry import Point, Polygon
# from graph_analysis import dict_data
graph_file = r'C:\Users\Ion\TFM\data\network_graphs'
study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
area = 'linthal'
new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
new_diG = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_DiGraph_largest.gpickle')
shp_path = str(study_area_dir) + '/' + area
file = open(str(graph_file) + "/ch_nodes_dict2056.pkl", 'rb')
nodes_dict = pickle.load(file)

def straight(nodes_dict, new_G, nodes_to_it=list(new_G.nodes)):
    node_straightness = {}
    node_dist = {}
    n = 0
    # i = 429340218
    # j = 429340228
    for i in nodes_to_it:
        print(n)
        i_lonlat = Point(nodes_dict[str(i)])
        dist_comp_list = []
        m = 0
        for j in list(new_G.nodes):
            m += 1
            # print('\r', m, end='')
            try:
                dist_comp = node_dist[(j, i)]
            except:
                if i == j:
                    continue
                j_lonlat = Point(nodes_dict[str(j)])
                eucl_dist = i_lonlat.distance(j_lonlat)
                sp_dist = nx.dijkstra_path_length(new_G, i, j, weight='length')
                dist_comp = eucl_dist / sp_dist
                node_dist[(i, j)] = dist_comp
            dist_comp_list.append(dist_comp)

        straightness = (1 / (len(list(new_G.nodes))-1)) * sum(dist_comp_list)
        node_straightness[i] = straightness
        n += 1
    return node_straightness

def dict_data(dicti, shp_path, attrib_name):
    min_val = min(dicti.values())
    max_val = max(dicti.values())
    avg_val = sum(dicti.values()) / len(dicti)
    len_val = len(dicti)
    with open(str(shp_path) + '/attr_' + str(attrib_name) + '.pkl', 'wb') as f:
        pickle.dump(dicti, f, pickle.HIGHEST_PROTOCOL)
    print(str(attrib_name) + ' (len,min,max,avg): ' +
          str(len_val) + ', ' + str(min_val) + ', ' + str(max_val) + ', ' + str(avg_val))
    return str([len_val, min_val,max_val,avg_val])

node_straightness = straight(nodes_dict, new_G)
attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
node_straightness_data = dict_data(node_straightness, shp_path, 'node_straightness')
attr_df.at['node_straightness', area] = node_straightness_data
attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])

# split list of node ids into bunches for multiprocessing
# num_procs = 32
# test_list = list(new_G.nodes)
# # initializing split index list
# split_list = []
# size = round(len(list(new_G.nodes))/num_procs)
# for i in range(1, num_procs):
#     idx = size * i
#     split_list.append(idx)
# res = [test_list[i: j] for i, j in zip([0] + split_list, split_list + [None])]
#
# # procs = [threading.Thread(target=straight, args=(nodes_dict, new_G, res, )) for _ in range(num_procs)]
# procs = []
# for i in range(0,num_procs):
#     res_v = res[i]
#     procs.append([threading.Thread(target=straight, args=(nodes_dict, new_G, res_v))])
#
# for proc in procs:
#     proc[0].start()
# for proc in procs:
#     proc[0].join()

# process1 = threading.Thread(target=straight, args=(nodes_dict, new_G, res[0]))
# process2 = threading.Thread(target=straight, args=(nodes_dict, new_G, res[1]))
# process3 = threading.Thread(target=straight, args=(nodes_dict, new_G, res[2]))
# process4 = threading.Thread(target=straight, args=(nodes_dict, new_G, res[3]))
# process5 = threading.Thread(target=straight, args=(nodes_dict, new_G, res[4]))
# process6 = threading.Thread(target=straight, args=(nodes_dict, new_G, res[5]))
# process7 = threading.Thread(target=straight, args=(nodes_dict, new_G, res[6]))
# process8 = threading.Thread(target=straight, args=(nodes_dict, new_G, res[7]))
#
# process1.start()
# process2.start()
# process3.start()
# process4.start()
# process5.start()
# process6.start()
# process7.start()
# process8.start()
#
# process1.join()
# process2.join()
# process3.join()
# process4.join()
# process5.join()
# process6.join()
# process7.join()
# process8.join()