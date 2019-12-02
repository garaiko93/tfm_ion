import pandas as pd
import pickle
import geopandas as gpd
import shapely.geometry as geo
import networkx as nx
import copy
import os
from shapely.geometry import Point, Polygon
import datetime

from network_graph import check_iso_graph


def dict_data(dicti, shp_path, attrib_name):
    min_val = min(dicti.values())
    max_val = max(dicti.values())
    avg_val = sum(dicti.values()) / len(dicti)
    len_val = len(dicti)
    with open(str(shp_path) + '/attr_' + str(attrib_name) + '.pkl', 'wb') as f:
        pickle.dump(dicti, f, pickle.HIGHEST_PROTOCOL)
    print(str(attrib_name) + ' (len,min,max,avg): ' +
          str(len_val) + ', ' + str(min_val) + ', ' + str(max_val) + ', ' + str(avg_val))
    return [len_val, min_val,max_val,avg_val]

def cut_graph(G,shp_path, graphtype, area, study_area_shp):
    new_G = copy.deepcopy(G)
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
    G = nx.read_gpickle(str(graph_file)+'\\ch_MultiDiGraph_bytime_largest.gpickle')
    diG = nx.read_gpickle(str(graph_file) + '\\ch_DiGraph_bytime_largest.gpickle')
    file = open(str(graph_file) + "\\ch_nodes_dict2056.pkl", 'rb')
    nodes_dict = pickle.load(file)
    print('----------------------------------------------------------------------')

    # Check if .csv with attributes exists:
    if os.path.isfile(str(study_area_dir) + '/' + 'attribute_table.csv') == True:
        attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",")
        print('Attr_df already exists, file loaded')
    else:
        attr_df = pd.DataFrame(data=None, columns=['area', 'n_nodes', 'n_edges', 'avg_degree', 'degree_centrality',
                                                   'avg_degree_connectivity', 'avg_edge_density',
                                                   'avg_shortest_path_duration', 'nodes_betweenness*',
                                                   'edges_betweenness', 'node_load_centrality', 'edge_load_centrality',
                                                   'clustering*', 'eccentricity', 'radius', 'diameter', 'center_nodes',
                                                   'periphery_nodes', 'barycenter_nodes'])
        print('Attributes_table does not exist, attr_df created empty')
    print('----------------------------------------------------------------------')

    study_area_list = list(os.walk(study_area_dir))[0][1]
    for area in study_area_list:
        print(area, datetime.datetime.now().time())
        shp_path = str(study_area_dir) + "/" + area
        study_area_shp = gpd.read_file(str(shp_path) + "/" + area + ".shp").iloc[0]['geometry']
        # Check if this area was filtered already by checking existance of done.txt
        if os.path.isfile(str(shp_path) + "/" + area + "_MultiDiGraph_largest.gpickle") == True:
            if attr_df['area'].str.contains(area).any() == False:
                print('Calculating topology attributes, graph already exists')
                attr_df = topology_attributes(study_area_dir, area, attr_df)
                continue
            else:
                print('This area was already filtered and attributes calculated')
                print('----------------------------------------------------------------------')
                continue

        new_G = cut_graph(G, shp_path, 'MultiDiGraph', area, study_area_shp)
        new_diG = cut_graph(diG, shp_path, 'DiGraph', area, study_area_shp)

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

        # Calculate attributes for new areas that has now the graph filtered
        attr_df = topology_attributes(study_area_dir, area, attr_df)
        print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('Process finished correctly: shp and graph files created in destination')

# NETWORK ANALYSIS ATTRIBUTES
def topology_attributes(study_area_dir, area, attr_df):
    # study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
    # area = 'zurich_small'
    new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
    new_diG = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_DiGraph_largest.gpickle')
    shp_path = str(study_area_dir) + '/' + area

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
    if os.path.isfile(str(study_area_dir) + '/' + 'attr_degree_centrality.csv') == False:
        degree_centr = nx.algorithms.degree_centrality(new_G)
    else:
        file = open(str(study_area_dir) + "\\attr_degree_centrality.pkl", 'rb')
        degree_centr = pickle.load(file)
    degree_centr_data = dict_data(degree_centr, shp_path, 'degree_centrality')

    # ----------------------------------------------------------------------
    # 3. AVG DEGREE OF CONECTIVITY: For a node of degree k - What is the average of its neighbours' degree?
    avg_degree_connect = nx.average_degree_connectivity(new_G)
    if os.path.isfile(str(study_area_dir) + '/' + 'attr_avg_degree_connect.csv') == False:
        avg_degree_connect = nx.average_degree_connectivity(new_G)
    else:
        file = open(str(study_area_dir) + "/attr_avg_degree_connect.pkl", 'rb')
        avg_degree_connect = pickle.load(file)
    avg_degree_connect_data = dict_data(avg_degree_connect, shp_path, 'avg_degree_connect')

    # ----------------------------------------------------------------------
    # 4. AVG EDGE DENSITY: Average edge density of the Graphs
    avg_edge_density = nx.density(new_G)
    # CHECK IF CELL HAS VALUE
    print('Average edge density: ' + str(avg_edge_density))

    # ----------------------------------------------------------------------
    # 5. AVG SHORTEST PATH LENGTH (WEIGHTED BY TIME):
    avg_spl = nx.average_shortest_path_length(new_G, weight='time')

    print('Average shortest path length (duration in seconds): ' + str(avg_spl))

    # ----------------------------------------------------------------------
    # 6. BETWEENNESS CENTRALITY: how many times a node or edge is passed for the sp
    # nodes betweenness
    node_betw_centr = nx.betweenness_centrality(new_diG, weight='time')
    node_betw_centr_data = dict_data(node_betw_centr, shp_path, 'node_betweenness')
    # edges betweenness
    edge_betw_centr = nx.edge_betweenness_centrality(new_G, weight='time')
    edge_betw_centr_data = dict_data(edge_betw_centr, shp_path, 'edge_betweenness')

    # ----------------------------------------------------------------------
    # 7. LOAD CENTRALITY:counts the number of shortest paths which cross each node/edge
    # nodes load:
    load_centrality = nx.load_centrality(new_G)
    load_centrality_data = dict_data(load_centrality, shp_path, 'node_load_centrality')
    # edges load:
    edge_load = nx.edge_load_centrality(new_G)
    edge_load_data = dict_data(edge_load, shp_path, 'edge_load_centrality')

    # ----------------------------------------------------------------------
    # 8. CLUSTERING: geometric average of the subgraph edge weights
    clustering = nx.clustering(new_diG)
    clustering_data = dict_data(clustering, shp_path, 'clustering')

    # ----------------------------------------------------------------------
    # NETWORK SHAPE ATTRIBUTES
    # 9. EXCENTRICITY: maximum distance from v to all other nodes in G
    # Radius/Diameter of graph: radius is minimum eccentricity/The diameter is the maximum eccentricity.
    eccentricity = nx.algorithms.distance_measures.eccentricity(new_G)
    eccentricity_data = dict_data(eccentricity, shp_path, 'eccentricity')
    radius = eccentricity_data[1]
    diameter = eccentricity_data[2]

    # Center: center is the set of nodes with eccentricity equal to radius.
    center = nx.algorithms.distance_measures.center(new_G)
    print('Center nodes: ' + str(center))
    # Periphery: set of nodes with eccentricity equal to the diameter.
    periphery = nx.algorithms.distance_measures.periphery(new_G)
    print('Periphery nodes: ' + str(periphery))
    # Baryecnter: subgraph that minimizes the function sum(w(u,v))
    barycenter = nx.algorithms.distance_measures.barycenter(new_G, weight='time')
    print('Baryenter nodes: ' + str(barycenter))
    # ----------------------------------------------------------------------
    # 10.

    # Create shp file with final graph
    new_row = {'area': area,
               'n_nodes': len(list(new_G)),
               'n_edges': len(new_G.edges()),
               'avg_degree': avg_deg,
               'degree_centrality': degree_centr_data,
               'avg_degree_connectivity': avg_degree_connect_data,
               'avg_edge_density': avg_edge_density,
               'avg_shortest_path_duration': avg_spl,
               'nodes_betweenness*': node_betw_centr_data,
               'edges_betweenness': edge_betw_centr_data,
               'node_load_centrality': load_centrality_data,
               'edge_load_centrality': edge_load_data,
               'clustering*': clustering_data,
               'eccentricity': eccentricity_data,
               'radius': radius,
               'diameter': diameter,
               'center_nodes': center,
               'periphery_nodes': periphery,
               'barycenter_nodes': barycenter
               # 'geometry': study_area_shp
               }
    attr_df = attr_df.append(new_row, ignore_index=True)
    attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=None)
    # attr_gdf = gpd.GeoDataFrame(attr_df)
    # attr_gdf[['area', 'geometry']].to_file(str(study_area_dir) + "\\attribute_table.shp")
    print('Attributes saved in attribute_table succesfully')
    print('----------------------------------------------------------------------')
    return attr_df


# study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
# topology_attributes(study_area_dir)
    # ideas for attributes:
    # concentration of nodes/edges in different grids
    # avg trip distance
    # distance of cg from home cluster to work/education cluster
    # take formulas from documentation (reference)
