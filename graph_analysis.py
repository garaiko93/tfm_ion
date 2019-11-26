import pandas as pd
import pickle
import geopandas as gpd
import shapely.geometry as geo
import networkx as nx
import copy
import os
from shapely.geometry import Point

from network_graph import check_iso_graph

def cut_graph(G,shp_path, graphtype,area):
    new_G = copy.deepcopy(G)
    study_area_shp = gpd.read_file(str(shp_path) + "/" + area + ".shp").iloc[0]['geometry']
    for node in list(new_G.nodes):
        point = Point(G.nodes[node]['x'], G.nodes[node]['y'])
        in_shp = study_area_shp.contains(point)
        if not in_shp:
            new_G.remove_node(node)
    filename =  str(area) + '_' + str(graphtype)
    [new_G, isolated, largest] = check_iso_graph(new_G, shp_path,filename)
    return new_G

# Filter graph and create a new one with nodes only into the shp
def filter_graph(study_area_dir, graph_file):
    print('Filtering graph process starts')
    # IMPORT data: graph and nodes_dict:
    G = nx.read_gpickle(str(graph_file)+'\ch_MultiDiGraph_bytime_largest.gpickle')
    G_s = nx.read_gpickle(str(graph_file) + '\ch_DiGraph_bytime_largest.gpickle')
    file = open(str(graph_file) + "\ch_nodes_dict2056.pkl", 'rb')
    nodes_dict = pickle.load(file)

    study_area_list = list(os.walk(study_area_dir))[0][1]

    print('----------------------------------------------------------------------')
    for area in study_area_list:
        print(area)
        shp_path = str(study_area_dir) + "/" + area

        new_G = cut_graph(G, shp_path, 'MultiDiGraph',area)
        new_diG = cut_graph(G_s, shp_path, 'DiGraph',area)

        # Create shp file with final graph
        df = pd.DataFrame(data=None, columns=['way_id', 'start_node_id', 'end_node_id', 'geometry'])
        for start, end, dup_way in list(new_G.edges):
            new_row = {'way_id': new_G[start][end][dup_way]['way_id'],
                       'start_node_id': start,
                       'end_node_id': end,
                       'geometry': geo.LineString([nodes_dict[start], nodes_dict[end]])
                       }
            df = df.append(new_row, ignore_index=True)
        gdf = gpd.GeoDataFrame(df)
        gdf.to_file(str(shp_path) + "/" + str(area) + "_network" + ".shp")
        print('Process finished correctly: shp and graph files created in destination')

        # NETWORK ANALYSIS ATTRIBUTES
        # degree = nx.degree(new_G).values()
        # print('Avg degree: ' +str(sum(degree)/len(degree)))
        # print('Degree centrality: ' + str(nx.algorithms.degree_centrality(new_G)))
        # print('Average edge density of the Graph: ' + str(nx.density(new_G)))  # Average edge density of the Graphs
        # print('Average shortest path length: ' + str(nx.average_shortest_path_length(new_G, weight='time')))  # Average shortest path length for ALL paths in the Graph
        # print('Average of its neighbours degree: ' + str(nx.average_degree_connectivity(new_G))) # For a node of degree k - What is the average of its neighbours' degree?

        # ideas for attributes:
        # concentration of nodes/edges in different grids
        # avg trip distance
        # distance of cg from home cluster to work/education cluster
        # take formulas from documentation (reference)
        print('----------------------------------------------------------------------')

    # analyse graph
    new_G = nx.read_gpickle(r'C:\Users\Ion\TFM\data\study_areas\zurich\zurich_MultiDiGraph_largest.gpickle')
    new_diG = nx.read_gpickle(r'C:\Users\Ion\TFM\data\study_areas\zurich\zurich_DiGraph_largest.gpickle')
    shp_path = r'C:\Users\Ion\TFM\data\study_areas\zurich'

    def dict_data(dicti, attrib_name):
        min_val = min(dicti.values())
        max_val = max(dicti.values())
        avg_val = sum(dicti.values()) / len(dicti)
        len_val = len(dicti)
        with open(str(shp_path) + '/attr_' + str(attrib_name) + '.pkl', 'wb') as f:
            pickle.dump(dicti, f, pickle.HIGHEST_PROTOCOL)
        print(str(attrib_name) + ' (len,min,max,avg): ' +
              str(len_val) + ', ' + str(min_val) + ', ' + str(max_val) + ', ' + str(avg_val))

    # ----------------------------------------------------------------------
    # 1. AVG DEGREE OF GRAPH: number of edges adjacent per node
    degree = nx.degree(new_G)
    degree_list = list(degree)
    sum_deg = 0
    count_deg = 0
    for node, deg in degree_list:
        sum_deg += deg
        count_deg += 1
    avg_deg = sum_deg/count_deg
    print('Average degree(edges adj per node): ' + str(avg_deg))

    # ----------------------------------------------------------------------
    # 2. DEGREE OF CENTRALITY:
    degree_centr = nx.algorithms.degree_centrality(new_G)
    dict_data(degree_centr, 'degree_centrality')

    # ----------------------------------------------------------------------
    # 3. AVG DEGREE OF CONECTIVITY: For a node of degree k - What is the average of its neighbours' degree?
    avg_degree_connect = nx.average_degree_connectivity(new_G)
    dict_data(avg_degree_connect, 'avg_degree_connect')

    # ----------------------------------------------------------------------
    # 4. AVG EDGE DENSITY: Average edge density of the Graphs
    avg_edge_density = nx.density(new_G)
    print('Average edge density (duration in seconds): ' + str(avg_edge_density))

    # ----------------------------------------------------------------------
    # 5. AVG SHORTEST PATH LENGTH (WEIGHTED BY TIME):
    avg_spl = nx.average_shortest_path_length(new_G, weight='time')
    print('Average shortest path length (duration in seconds): ' + str(avg_spl))

    # ----------------------------------------------------------------------
    # 6. BETWEENNESS CENTRALITY: how many times a node or edge is passed for the sp
    # nodes betweenness
    node_betw_centr = nx.betweenness_centrality(new_diG, weight = 'time')
    dict_data(node_betw_centr, 'node_betweenness')
    # edges betweenness
    edge_betw_centr = nx.edge_betweenness_centrality(new_G, weight = 'time')
    dict_data(edge_betw_centr, 'edge_betweenness')

    # ----------------------------------------------------------------------
    # 7. LOAD CENTRALITY:counts the number of shortest paths which cross each node/edge
    # nodes load:
    load_centrality = nx.load_centrality(new_G)
    dict_data(load_centrality, 'node_load_centrality')
    # edges load:
    edge_load = nx.edge_load_centrality(new_G)
    dict_data(edge_load, 'edge_load_centrality')

    # ----------------------------------------------------------------------
    # 8. CLUSTERING: geometric average of the subgraph edge weights
    clustering = nx.clustering(new_diG)
    dict_data(clustering, 'clustering')




