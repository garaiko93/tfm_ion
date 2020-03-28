import pandas as pd
import pickle
import geopandas as gpd
import networkx as nx
import os
import osmnx as ox
import datetime

# functions from other scripts
from acc_btw import dict_data, btw_acc, straightness, cut_graph, take_avg, save_attr_df, cutting_graph

def analysis_setup(study_area_dir, graph_file, area_def):
    print(datetime.datetime.now(), 'Area analysis starts.')
    print('----------------------------------------------------------------------')

    if area_def:
        filter_graph(study_area_dir, graph_file, area_def)
    else:
        study_area_list = list(os.walk(study_area_dir))[0][1]
        for area in study_area_list:
            filter_graph(study_area_dir, graph_file, area)


def filter_graph(study_area_dir, graph_file, area):
    print(datetime.datetime.now(), area)

    # Check if this area was filtered already by checking existance of _network_5k.shp
    if not os.path.isfile(str(study_area_dir) + "/" + str(area) + "/" + str(area) + "_network_5k.shp"):
        # Cuts swiss graph for study area, and 5k buffer area network
        cutting_graph(study_area_dir, graph_file, area)
    else:
        print(datetime.datetime.now(), 'Graph already exists')
    print('----------------------------------------------------------------------')

    # Calculate attributes for new areas that has now the graph filtered
    area_series, attributes = topology_attributes(study_area_dir, graph_file, area)
    # Manipulate attributes table
    take_avg(study_area_dir, area, attributes)
    print('----------------------------------------------------------------------')
    print(datetime.datetime.now(), 'Process finished correctly: shp and graph files created in output directory')


# NETWORK ANALYSIS ATTRIBUTES
def topology_attributes(study_area_dir, graph_file, area):
    # area = 'test_area'
    # graph_file = r'C:\Users\Ion\TFM\data\network_graphs'
    # study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'

    # Load necessary files
    file = open(str(graph_file) + "/ch_nodes_dict2056.pkl", 'rb')
    nodes_dict = pickle.load(file)

    shp_path = str(study_area_dir) + "/" + str(area)
    study_area_shp = gpd.read_file(str(shp_path) + "/" + str(area) + ".shp").iloc[0]['geometry']

    new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
    new_diG = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_DiGraph_largest.gpickle')
    new_G5k = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph5k_largest.gpickle')
    shp_path = str(study_area_dir) + '/' + area

    # Check if .csv with attributes exists:
    if os.path.isfile(str(study_area_dir) + '/' + 'attribute_table.csv'):
        attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
        print(datetime.datetime.now(), 'Attr_df already exists, file loaded')
    else:
        attr_df = pd.DataFrame(data=None)
        print(datetime.datetime.now(), 'Attributes_table does not exist, attr_df created empty')
    print('----------------------------------------------------------------------')

    # Define way_type for each area:
    area_type_dict = {'bern': 'urban',
                        'bern_large': 'urban',
                        'chur': 'rural',
                        'freiburg': 'rural',
                        'frutigen': 'mountain',
                        'lausanne': 'urban',
                        'lausanne_lake': 'urban',
                        'linthal': 'mountain',
                        'locarno': 'mountain',
                        'lugano': 'urban',
                        'luzern': 'urban',
                        'neuchatel': 'rural',
                        'plateau': 'rural',
                        'sion': 'mountain',
                        'stgallen': 'rural',
                        'test_area': 'urban',
                        'zermatt': 'mountain',
                        'zurich_kreis': 'urban',
                        'zurich_large': 'urban'}

    # check if table is up to date with all attributes:
    attributes = ['area_type',
                  'n_nodes',            # stats_basic
                  'n_edges',            # stats_basic
                  'network_distance',   # stats_basic
                  'area',
                  'population',
                  'trips',
                  'n_intersection',     # stats_basic
                  'n_street',           # stats_basic
                  'streets_per_node',   # stats_basic
                  'node_d_km',          # stats_basic
                  'intersection_d_km',  # stats_basic
                  'edge_d_km',          # stats_basic
                  'street_d_km',        # stats_basic
                  'circuity_avg',       # stats_basic
                  'avg_degree',
                  'avg_neighbor_degree',
                  'degree_centrality',
                  'avg_degree_connectivity',
                  'avg_edge_density',
                  'avg_shortest_path_duration',
                  'node_betweenness*',
                  'edge_betweenness',
                  # 'lim_edge_betweenness',
                  'btw_home_trip_production',
                  'btw_empl_trip_generation',
                  'btw_acc_trip_generation',
                  'btw_acc_trip_production',
                  'node_straightness',
                  'node_closeness_time*',
                  'node_closeness_length*',
                  'node_load_centrality',
                  'edge_load_centrality',
                  'clustering*',
                  'clustering_w*',
                  'eccentricity',
                  'radius',
                  'diameter',
                  'center_nodes',
                  'periphery_nodes',
                  'barycenter_nodes']

    # Update attribute table with new added attributes or study areas
    for attribute in attributes:
        if attribute not in attr_df.index:
            s = pd.Series(name=attribute)
            attr_df = attr_df.append(s)
            print(datetime.datetime.now(), 'Added ' + str(attribute) + ' as row to attr_df')

    # create empty row with areas name to add attributes
    if area not in attr_df.columns:
        new_column = pd.DataFrame({area: [len(list(new_G)), len(new_G.edges()), study_area_shp.area]},
                                  index=['n_nodes', 'n_edges', 'area'])
        attr_df = pd.concat([attr_df, new_column], axis=1, sort=False)
        attr_df = attr_df.astype({area: object})
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
        # save_attr_df(attr_df, study_area_dir, area)
        print(datetime.datetime.now(), 'Added ' + str(area) + ' as column to attr_df as dtype: ' +
              str(attr_df[area].dtype))
    print('----------------------------------------------------------------------')

    # Extract areas column as pandas Series, to permit parallel work with the attribute table
    area_series = attr_df[area]
    print(type(area_series), area_series.name)
    print(datetime.datetime.now(), 'Areas series extracted, ready to perform attributes of graph.')

    # ----------------------------------------------------------------------
    # 0.Basic information of study area
    if pd.isnull(area_series['area']):
    # if pd.isnull(attr_df.loc['area', area]):
        area_val = study_area_shp.area
        area_series = dict_data(area_val, study_area_dir, 'area', area, area_series)
    if pd.isnull(area_series['area_type']):
    # if pd.isnull(attr_df.loc['area_type', area]):
        area_series = dict_data(area_type_dict[area], study_area_dir, 'area_type', area, area_series)

    # ----------------------------------------------------------------------
    # 1. AVG DEGREE OF GRAPH: number of edges adjacent per node
    if pd.isnull(area_series['avg_degree']):
    # if pd.isnull(attr_df.loc['avg_degree', area]):
        degree = nx.degree(new_G)
        degree_list = list(degree)
        sum_deg = 0
        count_deg = 0
        for node, deg in degree_list:
            sum_deg += deg
            count_deg += 1
        avg_deg = sum_deg/count_deg
        area_series = dict_data(avg_deg, study_area_dir, 'avg_degree', area, area_series)
        # attr_df.at['avg_degree', area] = avg_deg
        # save_attr_df(attr_df, study_area_dir)
        # print(datetime.datetime.now(), 'Average degree(edges adj per node): ' + str(avg_deg))

    # avg_neighbour_degree:  average degree of the neighborhood of each node
    if pd.isnull(area_series['avg_neighbor_degree']):
    # if pd.isnull(attr_df.loc['avg_neighbor_degree', area]):
        avg_neighbor_degree = nx.average_neighbor_degree(new_G)
        area_series = dict_data(avg_neighbor_degree, study_area_dir, 'avg_neighbor_degree', area, area_series)
        # avg_neighbor_degree_data = dict_data(avg_neighbor_degree, shp_path, 'avg_neighbor_degree')
        # attr_df.at['avg_neighbor_degree', area] = avg_neighbor_degree_data
        # save_attr_df(attr_df, study_area_dir)
    # ----------------------------------------------------------------------
    # 2. DEGREE OF CENTRALITY: The degree centrality for a node v is the fraction of nodes it is connected to (normalized)
    if pd.isnull(area_series['degree_centrality']):
    # if attr_df.loc['degree_centrality', area] is None:
        degree_centr = nx.algorithms.degree_centrality(new_G)
        area_series = dict_data(degree_centr, study_area_dir, 'degree_centrality', area, area_series)

        # degree_centr_data = dict_data(degree_centr, shp_path, 'degree_centrality')
        # attr_df.at['degree_centrality', area] = degree_centr_data
        # save_attr_df(attr_df, study_area_dir)
    # ----------------------------------------------------------------------
    # 3. CONNECTIVITY:  is the average nearest neighbor degree of nodes with degree k
    if pd.isnull(area_series['avg_degree_connectivity']):
    # if pd.isnull(attr_df.loc['avg_degree_connectivity', area]):
        avg_degree_connect = nx.average_degree_connectivity(new_G)
        area_series = dict_data(avg_degree_connect, study_area_dir, 'avg_degree_connectivity', area, area_series)
        # avg_degree_connect_data = dict_data(avg_degree_connect, shp_path, 'avg_degree_connect')
        # attr_df.at['avg_degree_connectivity', area] = avg_degree_connect_data
        # save_attr_df(attr_df, study_area_dir)
    # if pd.isnull(attr_df.loc['node_connectivity', area]) == True:
    #     node_connect = nx.node_connectivity(new_G)
    #     attr_df.at['node_connectivity', area] = node_connect
    #     print('Node connectivity k: ' + str(node_connect))
        # edge_connect = nx.edge_connectivity(new_G)
        # attr_df.at['edge_connectivity', area] = edge_connect
        # print('Node connectivity k: ' + str(edge_connect))
        # avg_node_connect =nx.average_node_connectivity(new_G)
        # attr_df.at['avg_node_connectivity', area] = avg_node_connect
        # print('Node connectivity k: ' + str(avg_node_connect))
        # attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # ----------------------------------------------------------------------
    # 4. AVG EDGE DENSITY: Average edge density of the Graphs.
    # The density is 0 for a graph without edges and 1 for a complete graph.
    if pd.isnull(area_series['avg_edge_density']):
    # if pd.isnull(attr_df.loc['avg_edge_density', area]):
        avg_edge_density = nx.density(new_G)
        area_series = dict_data(avg_edge_density, study_area_dir, 'avg_edge_density', area, area_series)
        # attr_df.at['avg_edge_density', area] = avg_edge_density
        # save_attr_df(attr_df, study_area_dir)
        # print(datetime.datetime.now(), 'Average edge density: ' + str(avg_edge_density))

    # ----------------------------------------------------------------------
    # 5. AVG SHORTEST PATH LENGTH (WEIGHTED BY TIME):
    if pd.isnull(area_series['avg_shortest_path_duration']):
    # if pd.isnull(attr_df.loc['avg_shortest_path_duration', area]):
        avg_spl = nx.average_shortest_path_length(new_G, weight='time')
        area_series = dict_data(avg_spl, study_area_dir, 'avg_shortest_path_duration', area, area_series)
        # attr_df.at['avg_shortest_path_duration', area] = avg_spl
        # save_attr_df(attr_df, study_area_dir)
        # print(datetime.datetime.now(), 'Average shortest path length (duration in seconds): ' + str(avg_spl))
    # ----------------------------------------------------------------------
    # 6. BETWEENNESS CENTRALITY: the fraction of nodes/edges of how many times is passed for the sp
    # nodes betweenness: not agree with the result of this algorithm based on the definition of betweenness...
    if pd.isnull(area_series['node_betweenness*']):
    # if pd.isnull(attr_df.loc['node_betweenness*', area]):
        node_betw_centr = nx.betweenness_centrality(new_diG, weight='time')
        area_series = dict_data(node_betw_centr, study_area_dir, 'node_betweenness*', area, area_series)
        # node_betw_centr_data = dict_data(node_betw_centr, shp_path, 'node_betweenness')
        # attr_df.at['node_betweenness*', area] = node_betw_centr_data
        # save_attr_df(attr_df, study_area_dir)
    # edges betweenness
    if pd.isnull(area_series['edge_betweenness']):
    # if pd.isnull(attr_df.loc['edge_betweenness', area]):
        edge_betw_centr = nx.edge_betweenness_centrality(new_G, weight='time')
        area_series = dict_data(edge_betw_centr, study_area_dir, 'edge_betweenness', area, area_series)
        # edge_betw_centr_data = dict_data(edge_betw_centr, shp_path, 'edge_betweenness')
        # attr_df.at['edge_betweenness', area] = edge_betw_centr_data
        # save_attr_df(attr_df, study_area_dir)
    if pd.isnull(area_series['btw_acc_trip_production']):
    # if pd.isnull(attr_df.loc['btw_acc_trip_production', area]):
        print('----------------------------------------------------------------------')
        btw_acc(new_G, new_G5k, study_area_dir, area, nodes_dict, area_series)
        print('----------------------------------------------------------------------')
        # dict_data(lim_edge_betw_centr, shp_path, 'lim_edge_betweenness*', area, attr_df)
        # lim_edge_betw_centr_data = dict_data(lim_edge_betw_centr, shp_path, 'lim_edge_betweenness')
        # attr_df.at['lim_edge_betweenness', area] = lim_edge_betw_centr_data
        # save_attr_df(attr_df, study_area_dir)
    # ----------------------------------------------------------------------
    # 7. CLOSENESS CENTRALITY: Of a node is the average length of the shortest path from the node to all other nodes
    if pd.isnull(area_series['node_closeness_time*']):
    # if pd.isnull(attr_df.loc['node_closeness_time*', area]):
        node_close_time_centr = nx.closeness_centrality(new_diG, distance='time')
        area_series = dict_data(node_close_time_centr, study_area_dir, 'node_closeness_time*', area, area_series)
        # node_close_centr_data = dict_data(node_close_time_centr, shp_path, 'node_closeness_time')
        # attr_df.at['node_closeness_time*', area] = node_close_centr_data
        # save_attr_df(attr_df, study_area_dir)
    if pd.isnull(area_series['node_closeness_length*']):
    # if pd.isnull(attr_df.loc['node_closeness_length*', area]):
        node_close_dist_centr = nx.closeness_centrality(new_diG, distance='length')
        area_series = dict_data(node_close_dist_centr, study_area_dir, 'node_closeness_length*', area, area_series)
        # node_close_dist_data = dict_data(node_close_dist_centr, shp_path, 'node_closeness_length')
        # attr_df.at['node_closeness_length*', area] = node_close_dist_data
        # save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # 8. STRAIGHTNESS CENTRALITY: compares the shortest path with the euclidean distance of each pair of nodes
    if pd.isnull(area_series['node_straightness']):
    # if pd.isnull(attr_df.loc['node_straightness', area]):
        print('----------------------------------------------------------------------')
        node_straightness = straightness(nodes_dict, new_G)
        area_series = dict_data(node_straightness, study_area_dir, 'node_straightness', area, area_series)
        print('----------------------------------------------------------------------')
        # attr_df.at['node_straightness', area] = node_straightness_data
        # save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # 8. LOAD CENTRALITY:counts the fraction of shortest paths which cross each node/edge
    # nodes load: of a node is the fraction of all shortest paths that pass through that node.
    if pd.isnull(area_series['node_load_centrality']):
    # if pd.isnull(attr_df.loc['node_load_centrality', area]):
        load_centrality = nx.load_centrality(new_G)
        area_series = dict_data(load_centrality, study_area_dir, 'node_load_centrality', area, area_series)
        # load_centrality_data = dict_data(load_centrality, shp_path, 'node_load_centrality')
        # attr_df.at['node_load_centrality', area] = load_centrality_data
        # save_attr_df(attr_df, study_area_dir)
    # edges load: counts the number of shortest paths which cross each edge
    if pd.isnull(area_series['edge_load_centrality']):
    # if pd.isnull(attr_df.loc['edge_load_centrality', area]):
        edge_load = nx.edge_load_centrality(new_G)
        area_series = dict_data(edge_load, study_area_dir, 'edge_load_centrality', area, area_series)
        # edge_load_data = dict_data(edge_load, shp_path, 'edge_load_centrality')
        # attr_df.at['edge_load_centrality', area] = edge_load_data
        # save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # 9. CLUSTERING: geometric average of the subgraph edge weights
    if pd.isnull(area_series['clustering_w*']):
    # if pd.isnull(attr_df.loc['clustering*', area]):
        clustering = nx.clustering(new_diG)
        area_series = dict_data(clustering, study_area_dir, 'clustering*', area, area_series)
        # clustering_data = dict_data(clustering, shp_path, 'clustering')
        # attr_df.at['clustering*', area] = clustering_data

        clustering_weighted = nx.clustering(new_diG, weight='time')
        area_series = dict_data(clustering_weighted, study_area_dir, 'clustering_w*', area, area_series)
        # clustering_weighted_data = dict_data(clustering_weighted, shp_path, 'clustering_w')
        # attr_df.at['clustering_w*', area] = clustering_weighted_data

        # save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # NETWORK SHAPE ATTRIBUTES
    # 10. ECCENTRICITY: maximum distance from v to all other nodes in G
    # Radius/Diameter of graph: radius is minimum eccentricity/The diameter is the maximum eccentricity.
    if pd.isnull(area_series['eccentricity']):
    # if pd.isnull(attr_df.loc['eccentricity', area]):
        eccentricity = nx.algorithms.distance_measures.eccentricity(new_G)
        area_series = dict_data(eccentricity, study_area_dir, 'eccentricity', area, area_series)
        area_series = dict_data(min(eccentricity.values()), study_area_dir, 'radius', area, area_series)
        area_series = dict_data(max(eccentricity.values()), study_area_dir, 'diameter', area, area_series)
        # eccentricity_data = dict_data(eccentricity, shp_path, 'eccentricity')
        # attr_df.at['eccentricity', area] = eccentricity_data
        # attr_df.at['radius', area] = nx.radius(new_G)
        # attr_df.at['diameter', area] = nx.diameter(new_G)
        # save_attr_df(attr_df, study_area_dir)

    # Center: center is the set of nodes with eccentricity equal to radius.
    # if pd.isnull(area_series['center_nodes']):
    # if pd.isnull(attr_df.loc['center_nodes', area]):
    #     center = nx.algorithms.distance_measures.center(new_G)
    #     attr_df.at['center_nodes', area] = center
    #     print(datetime.datetime.now(), 'Center nodes: ' + str(center))
    #     save_attr_df(attr_df, study_area_dir)
    # # Periphery: set of nodes with eccentricity equal to the diameter.
    # if pd.isnull(area_series['periphery_nodes']):
    # if pd.isnull(attr_df.loc['periphery_nodes', area]):
    #     periphery = nx.algorithms.distance_measures.periphery(new_G)
    #     attr_df.at['periphery_nodes', area] = periphery
    #     print(datetime.datetime.now(), 'Periphery nodes: ' + str(periphery))
    #     save_attr_df(attr_df, study_area_dir)
    # # Baryecnter: subgraph that minimizes the function sum(w(u,v))
    # if pd.isnull(area_series['barycenter_nodes']):
    # if pd.isnull(attr_df.loc['barycenter_nodes', area]):
    #     barycenter = nx.algorithms.distance_measures.barycenter(new_G, weight='time')
    #     attr_df.at['barycenter_nodes', area] = barycenter
    #     print(datetime.datetime.now(), 'Baryenter nodes: ' + str(barycenter))
    #     save_attr_df(attr_df, study_area_dir)

    # 'network_distance'
    if pd.isnull(area_series['network_distance']):
    # if pd.isnull(attr_df.loc['network_distance', area]):
        nodes_list = new_G.edges.data('length', default=1)
        total_len = 0
        for nod in nodes_list:
            total_len += nod[2]

        area_series = dict_data(total_len, study_area_dir, 'network_distance', area, area_series)
        # attr_df.at['network_distance', area] = total_len
        # print(datetime.datetime.now(), 'Total network distance: ' + str(total_len))
        # save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # 11. OSMnx stats module
    if not os.path.isfile(str(shp_path) + '/' + 'stats_basic.pkl'):
        print(datetime.datetime.now(), 'Calculating basic_stats')
        new_G.graph['crs'] = 'epsg:2056'
        new_G.graph['name'] = str(area) + '_MultiDiGraph'
        basic_stats = ox.basic_stats(new_G, area=study_area_shp.area, clean_intersects=True, tolerance=15,
                                     circuity_dist='euclidean')
        with open(str(shp_path) + '/stats_basic.pkl', 'wb') as f:
            pickle.dump(basic_stats, f, pickle.HIGHEST_PROTOCOL)
    # if not os.path.isfile(str(study_area_dir) + '/' + 'stats_extended.pkl'):
        # print('Calculating extended_stats')
        # extended_stats = ox.extended_stats(new_G, connectivity=True, anc=True, ecc=True, bc=True, cc=True)
        # with open(str(shp_path) + '/stats_extended.pkl', 'wb') as f:
        #     pickle.dump(extended_stats, f, pickle.HIGHEST_PROTOCOL)

    # transbase of attributes from stats_basic dic to attributes table:
    basic_stats_list = {'n': 'n_nodes',
                        'm': 'n_edges',
                        'edge_length_total': 'network_distance',
                        'intersection_count': 'n_intersection',
                        'street_segments_count': 'n_street',
                        'streets_per_node_avg': 'streets_per_node',
                        'node_density_km': 'node_d_km',
                        'intersection_density_km': 'intersection_d_km',
                        'edge_density_km': 'edge_d_km',
                        'street_density_km': 'street_d_km',
                        'circuity_avg': 'circuity_avg'}
    for attr in basic_stats_list:
        if pd.isnull(area_series[basic_stats_list[attr]]):
        # if pd.isnull(attr_df.loc[basic_stats_list[attr], area]):
            file = open(str(shp_path) + "/stats_basic.pkl", 'rb')
            stats_basic_dict = pickle.load(file)
            area_series = dict_data(stats_basic_dict[attr], study_area_dir, basic_stats_list[attr], area, area_series)
            # attr_df.at[basic_stats_list[attr], area] = stats_basic_dict[attr]
            # print(datetime.datetime.now(), 'Attribute ' + str(basic_stats_list[attr]) + ': '
            #       + str(stats_basic_dict[attr]))
            # save_attr_df(attr_df, study_area_dir)
    print('----------------------------------------------------------------------')
    print(datetime.datetime.now(), 'All attributes were calculated successfully.')
    print('----------------------------------------------------------------------')

    return area_series, attributes

# Filter graph and create a new one with nodes only into the shp
# def filter_graph(study_area_dir, graph_file, area_def=None):
#     print(datetime.datetime.now(), 'Filtering graph process starts')
#     # IMPORT data: graph and nodes_dict:
#     file = open(str(graph_file) + "/ch_nodes_dict2056.pkl", 'rb')
#     nodes_dict = pickle.load(file)
#     print('----------------------------------------------------------------------')
#
#     # Check if .csv with attributes exists:
#     if os.path.isfile(str(study_area_dir) + '/' + 'attribute_table.csv'):
#         # study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
#         attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
#         print(datetime.datetime.now(), 'Attr_df already exists, file loaded')
#     else:
#         attr_df = pd.DataFrame(data=None)
#         print(datetime.datetime.now(), 'Attributes_table does not exist, attr_df created empty')
#
#     print('----------------------------------------------------------------------')
#
#     study_area_list = list(os.walk(study_area_dir))[0][1]
#     for area in study_area_list:
#         if area_def:
#             area = area_def
#         if area == 'chur':
#             continue
#         if area == 'freiburg':
#             continue
#         print(type(area))
#         print(datetime.datetime.now(), area)
#         shp_path = str(study_area_dir) + "/" + str(area)
#         study_area_shp = gpd.read_file(str(shp_path) + "/" + str(area) + ".shp").iloc[0]['geometry']
#
#         # Check if this area was filtered already by checking existance of done.txt
#         if os.path.isfile(str(shp_path) + "/" + str(area) + "_network_5k.shp"):
#             print(datetime.datetime.now(), 'Graph already exists')
#             print('----------------------------------------------------------------------')
#             attr_df, attributes = topology_attributes(study_area_dir, area, attr_df, study_area_shp, nodes_dict)
#             # Manipulate attributes table
#             take_avg(attr_df, study_area_dir, attributes)
#             if area_def:
#                 break
#         else:
#             print(datetime.datetime.now(), 'Creating graph for area ...')
#
#             G = nx.read_gpickle(str(graph_file) + '/ch_MultiDiGraph_bytime_largest.gpickle')
#             diG = nx.read_gpickle(str(graph_file) + '/ch_DiGraph_bytime_largest.gpickle')
#
#             new_G = cut_graph(G, shp_path, 'MultiDiGraph', area, study_area_shp)
#             new_diG = cut_graph(diG, shp_path, 'DiGraph', area, study_area_shp)
#
#             poly_line = LinearRing(study_area_shp.exterior)
#             poly_line_offset = poly_line.buffer(5000, resolution=16, join_style=2, mitre_limit=1)
#             study_area_shp5k = Polygon(list(poly_line_offset.exterior.coords))
#
#             new_G5k = cut_graph(G, shp_path, 'MultiDiGraph5k', area, study_area_shp5k)
#
#             # Create shp file with final graph
#             def export_shp(new_G, filename):
#                 print(datetime.datetime.now(), 'Creating shp file of network ...')
#                 df = pd.DataFrame(data=None, columns=['way_id', 'start_node_id', 'end_node_id', 'geometry'])
#                 for start, end, dup_way in list(new_G.edges):
#                     new_row = {'way_id': new_G[start][end][dup_way]['way_id'],
#                                'start_node_id': start,
#                                'end_node_id': end,
#                                'geometry': geo.LineString([nodes_dict[str(start)], nodes_dict[str(end)]])
#                                }
#                     df = df.append(new_row, ignore_index=True)
#                 gdf = gpd.GeoDataFrame(df)
#                 gdf.to_file(str(filename))
#
#             # Call function to export shp file
#             export_shp(new_G, str(shp_path) + "/" + str(area) + "_network.shp")
#             export_shp(new_G5k, str(shp_path) + "/" + str(area) + "_network_5k.shp")
#             print('----------------------------------------------------------------------')
#             # Calculate attributes for new areas that has now the graph filtered
#             attr_df, attributes = topology_attributes(study_area_dir, area, attr_df, study_area_shp, nodes_dict)
#             # Manipulate attributes table
#             take_avg(attr_df, study_area_dir, attributes)
#             if area_def:
#                 break
#             print('----------------------------------------------------------------------')
#     print('----------------------------------------------------------------------')
#     print(datetime.datetime.now(), 'Process finished correctly: shp and graph files created in output directory')


# NETWORK ANALYSIS ATTRIBUTES
# def topology_attributes2(study_area_dir, area, attr_df, study_area_shp, nodes_dict):
#     # graph_file = r'C:\Users\Ion\TFM\data\network_graphs\test'
#     # study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
#     # area = 'freiburg'
#
#     new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
#     new_diG = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_DiGraph_largest.gpickle')
#     new_G30k = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph5k_largest.gpickle')
#     shp_path = str(study_area_dir) + '/' + area
#
#     # Define way_type for each area:
#     area_type_dict = {'bern': 'urban',
#                         'bern_large': 'urban',
#                         'chur': 'rural',
#                         'freiburg': 'rural',
#                         'frutigen': 'mountain',
#                         'lausanne': 'urban',
#                         'lausanne_lake': 'urban',
#                         'linthal': 'mountain',
#                         'locarno': 'mountain',
#                         'lugano': 'urban',
#                         'luzern': 'urban',
#                         'neuchatel': 'rural',
#                         'plateau': 'rural',
#                         'sion': 'mountain',
#                         'stgallen': 'rural',
#                         'test_area': 'urban',
#                         'zermatt': 'mountain',
#                         'zurich_kreis': 'urban',
#                         'zurich_large': 'urban'}
#
#     # check if table is up to date with all attributes:
#     attributes = ['area_type',
#                   'n_nodes',            # stats_basic
#                   'n_edges',            # stats_basic
#                   'network_distance',   # stats_basic
#                   'area',
#                   'population',
#                   'trips',
#                   'n_intersection',     # stats_basic
#                   'n_street',           # stats_basic
#                   'streets_per_node',   # stats_basic
#                   'node_d_km',          # stats_basic
#                   'intersection_d_km',  # stats_basic
#                   'edge_d_km',          # stats_basic
#                   'street_d_km',        # stats_basic
#                   'circuity_avg',       # stats_basic
#                   'avg_degree',
#                   'avg_neighbor_degree',
#                   'degree_centrality',
#                   'avg_degree_connectivity',
#                   'avg_edge_density',
#                   'avg_shortest_path_duration',
#                   'node_betweenness*',
#                   'edge_betweenness',
#                   # 'lim_edge_betweenness',
#                   'btw_home_trip_production',
#                   'btw_empl_trip_generation',
#                   'btw_acc_trip_generation',
#                   'btw_acc_trip_production',
#                   'node_straightness',
#                   'node_closeness_time*',
#                   'node_closeness_length*',
#                   'node_load_centrality',
#                   'edge_load_centrality',
#                   'clustering*',
#                   'clustering_w*',
#                   'eccentricity',
#                   'radius',
#                   'diameter',
#                   'center_nodes',
#                   'periphery_nodes',
#                   'barycenter_nodes']
#
#     for attribute in attributes:
#         if attribute not in attr_df.index:
#             s = pd.Series(name=attribute)
#             attr_df = attr_df.append(s)
#             print(datetime.datetime.now(), 'Added ' + str(attribute) + ' as row to attr_df')
#
#     # create empty row with areas name to add attributes
#     if area not in attr_df.columns:
#         new_column = pd.DataFrame({area: [len(list(new_G)), len(new_G.edges()), study_area_shp.area]},
#                                   index=['n_nodes', 'n_edges', 'area'])
#         attr_df = pd.concat([attr_df, new_column], axis=1, sort=False)
#         attr_df = attr_df.astype({area: object})
#         print(datetime.datetime.now(), 'Added ' + str(area) + ' as column to attr_df as dtype: ' +
#               str(attr_df[area].dtype))
#         print('----------------------------------------------------------------------')
#     else:
#         print(datetime.datetime.now(), 'Area exists in attributes table, checking for additional attributes')
#         print('----------------------------------------------------------------------')
#
#     # ----------------------------------------------------------------------
#     # 0.Basic information of study area
#     if pd.isnull(attr_df.loc['area', area]):
#         area_val = study_area_shp.area
#         attr_df = dict_data(area_val, shp_path, 'area', area, attr_df)
#         # attr_df.at['area', area] = area_val
#         # save_attr_df(attr_df, study_area_dir)
#         # print(datetime.datetime.now(), 'Area of study area: ' + str(area_val))
#     if pd.isnull(attr_df.loc['area_type', area]):
#         attr_df = dict_data(area_type_dict[area], shp_path, 'area_type', area, attr_df)
#
#     # ----------------------------------------------------------------------
#     # 1. AVG DEGREE OF GRAPH: number of edges adjacent per node
#     if pd.isnull(attr_df.loc['avg_degree', area]):
#         degree = nx.degree(new_G)
#         degree_list = list(degree)
#         sum_deg = 0
#         count_deg = 0
#         for node, deg in degree_list:
#             sum_deg += deg
#             count_deg += 1
#         avg_deg = sum_deg/count_deg
#         attr_df = dict_data(avg_deg, shp_path, 'avg_degree', area, attr_df)
#         # attr_df.at['avg_degree', area] = avg_deg
#         # save_attr_df(attr_df, study_area_dir)
#         # print(datetime.datetime.now(), 'Average degree(edges adj per node): ' + str(avg_deg))
#
#     # avg_neighbour_degree:  average degree of the neighborhood of each node
#     if pd.isnull(attr_df.loc['avg_neighbor_degree', area]):
#         avg_neighbor_degree = nx.average_neighbor_degree(new_G)
#         attr_df = dict_data(avg_neighbor_degree, shp_path, 'avg_neighbor_degree', area, attr_df)
#         # avg_neighbor_degree_data = dict_data(avg_neighbor_degree, shp_path, 'avg_neighbor_degree')
#         # attr_df.at['avg_neighbor_degree', area] = avg_neighbor_degree_data
#         # save_attr_df(attr_df, study_area_dir)
#     # ----------------------------------------------------------------------
#     # 2. DEGREE OF CENTRALITY: The degree centrality for a node v is the fraction of nodes it is connected to (normalized)
#     if attr_df.loc['degree_centrality', area] is None:
#         degree_centr = nx.algorithms.degree_centrality(new_G)
#         attr_df = dict_data(degree_centr, shp_path, 'degree_centrality', area, attr_df)
#
#         # degree_centr_data = dict_data(degree_centr, shp_path, 'degree_centrality')
#         # attr_df.at['degree_centrality', area] = degree_centr_data
#         # save_attr_df(attr_df, study_area_dir)
#     # ----------------------------------------------------------------------
#     # 3. CONNECTIVITY:  is the average nearest neighbor degree of nodes with degree k
#     if pd.isnull(attr_df.loc['avg_degree_connectivity', area]):
#         avg_degree_connect = nx.average_degree_connectivity(new_G)
#         attr_df = dict_data(avg_degree_connect, shp_path, 'avg_degree_connectivity', area, attr_df)
#         # avg_degree_connect_data = dict_data(avg_degree_connect, shp_path, 'avg_degree_connect')
#         # attr_df.at['avg_degree_connectivity', area] = avg_degree_connect_data
#         # save_attr_df(attr_df, study_area_dir)
#     # if pd.isnull(attr_df.loc['node_connectivity', area]) == True:
#     #     node_connect = nx.node_connectivity(new_G)
#     #     attr_df.at['node_connectivity', area] = node_connect
#     #     print('Node connectivity k: ' + str(node_connect))
#         # edge_connect = nx.edge_connectivity(new_G)
#         # attr_df.at['edge_connectivity', area] = edge_connect
#         # print('Node connectivity k: ' + str(edge_connect))
#         # avg_node_connect =nx.average_node_connectivity(new_G)
#         # attr_df.at['avg_node_connectivity', area] = avg_node_connect
#         # print('Node connectivity k: ' + str(avg_node_connect))
#         # attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
#     # ----------------------------------------------------------------------
#     # 4. AVG EDGE DENSITY: Average edge density of the Graphs.
#     # The density is 0 for a graph without edges and 1 for a complete graph.
#     if pd.isnull(attr_df.loc['avg_edge_density', area]):
#         avg_edge_density = nx.density(new_G)
#         attr_df = dict_data(avg_edge_density, shp_path, 'avg_edge_density', area, attr_df)
#         # attr_df.at['avg_edge_density', area] = avg_edge_density
#         # save_attr_df(attr_df, study_area_dir)
#         # print(datetime.datetime.now(), 'Average edge density: ' + str(avg_edge_density))
#
#     # ----------------------------------------------------------------------
#     # 5. AVG SHORTEST PATH LENGTH (WEIGHTED BY TIME):
#     if pd.isnull(attr_df.loc['avg_shortest_path_duration', area]):
#         avg_spl = nx.average_shortest_path_length(new_G, weight='time')
#         attr_df = dict_data(avg_spl, shp_path, 'avg_shortest_path_duration', area, attr_df)
#         # attr_df.at['avg_shortest_path_duration', area] = avg_spl
#         # save_attr_df(attr_df, study_area_dir)
#         # print(datetime.datetime.now(), 'Average shortest path length (duration in seconds): ' + str(avg_spl))
#     # ----------------------------------------------------------------------
#     # 6. BETWEENNESS CENTRALITY: the fraction of nodes/edges of how many times is passed for the sp
#     # nodes betweenness: not agree with the result of this algorithm based on the definition of betweenness...
#     if pd.isnull(attr_df.loc['node_betweenness*', area]):
#         node_betw_centr = nx.betweenness_centrality(new_diG, weight='time')
#         attr_df = dict_data(node_betw_centr, shp_path, 'node_betweenness*', area, attr_df)
#         # node_betw_centr_data = dict_data(node_betw_centr, shp_path, 'node_betweenness')
#         # attr_df.at['node_betweenness*', area] = node_betw_centr_data
#         # save_attr_df(attr_df, study_area_dir)
#     # edges betweenness
#     if pd.isnull(attr_df.loc['edge_betweenness', area]):
#         edge_betw_centr = nx.edge_betweenness_centrality(new_G, weight='time')
#         attr_df = dict_data(edge_betw_centr, shp_path, 'edge_betweenness', area, attr_df)
#         # edge_betw_centr_data = dict_data(edge_betw_centr, shp_path, 'edge_betweenness')
#         # attr_df.at['edge_betweenness', area] = edge_betw_centr_data
#         # save_attr_df(attr_df, study_area_dir)
#     # if pd.isnull(attr_df.loc['btw_acc_trip_production', area]):
#     #     print('----------------------------------------------------------------------')
#     #     btw_acc(new_G, new_G30k, shp_path, area, nodes_dict, attr_df)
#     #     print('----------------------------------------------------------------------')
#         # dict_data(lim_edge_betw_centr, shp_path, 'lim_edge_betweenness*', area, attr_df)
#         # lim_edge_betw_centr_data = dict_data(lim_edge_betw_centr, shp_path, 'lim_edge_betweenness')
#         # attr_df.at['lim_edge_betweenness', area] = lim_edge_betw_centr_data
#         # save_attr_df(attr_df, study_area_dir)
#     # ----------------------------------------------------------------------
#     # 7. CLOSENESS CENTRALITY: Of a node is the average length of the shortest path from the node to all other nodes
#     if pd.isnull(attr_df.loc['node_closeness_time*', area]):
#         node_close_time_centr = nx.closeness_centrality(new_diG, distance='time')
#         attr_df = dict_data(node_close_time_centr, shp_path, 'node_closeness_time*', area, attr_df)
#         # node_close_centr_data = dict_data(node_close_time_centr, shp_path, 'node_closeness_time')
#         # attr_df.at['node_closeness_time*', area] = node_close_centr_data
#         # save_attr_df(attr_df, study_area_dir)
#     if pd.isnull(attr_df.loc['node_closeness_length*', area]):
#         node_close_dist_centr = nx.closeness_centrality(new_diG, distance='length')
#         attr_df = dict_data(node_close_dist_centr, shp_path, 'node_closeness_length*', area, attr_df)
#         # node_close_dist_data = dict_data(node_close_dist_centr, shp_path, 'node_closeness_length')
#         # attr_df.at['node_closeness_length*', area] = node_close_dist_data
#         # save_attr_df(attr_df, study_area_dir)
#
#     # ----------------------------------------------------------------------
#     # 8. STRAIGHTNESS CENTRALITY: compares the shortest path with the euclidean distance of each pair of nodes
#     if pd.isnull(attr_df.loc['node_straightness', area]):
#         print('----------------------------------------------------------------------')
#         node_straightness = straightness(nodes_dict, new_G)
#         attr_df = dict_data(node_straightness, shp_path, 'node_straightness', area, attr_df)
#         print('----------------------------------------------------------------------')
#         # attr_df.at['node_straightness', area] = node_straightness_data
#         # save_attr_df(attr_df, study_area_dir)
#
#     # ----------------------------------------------------------------------
#     # 8. LOAD CENTRALITY:counts the fraction of shortest paths which cross each node/edge
#     # nodes load: of a node is the fraction of all shortest paths that pass through that node.
#     if pd.isnull(attr_df.loc['node_load_centrality', area]):
#         load_centrality = nx.load_centrality(new_G)
#         attr_df = dict_data(load_centrality, shp_path, 'node_load_centrality', area, attr_df)
#         # load_centrality_data = dict_data(load_centrality, shp_path, 'node_load_centrality')
#         # attr_df.at['node_load_centrality', area] = load_centrality_data
#         # save_attr_df(attr_df, study_area_dir)
#     # edges load: counts the number of shortest paths which cross each edge
#     if pd.isnull(attr_df.loc['edge_load_centrality', area]):
#         edge_load = nx.edge_load_centrality(new_G)
#         attr_df = dict_data(edge_load, shp_path, 'edge_load_centrality', area, attr_df)
#         # edge_load_data = dict_data(edge_load, shp_path, 'edge_load_centrality')
#         # attr_df.at['edge_load_centrality', area] = edge_load_data
#         # save_attr_df(attr_df, study_area_dir)
#
#     # ----------------------------------------------------------------------
#     # 9. CLUSTERING: geometric average of the subgraph edge weights
#     if pd.isnull(attr_df.loc['clustering*', area]):
#         clustering = nx.clustering(new_diG)
#         attr_df = dict_data(clustering, shp_path, 'clustering*', area, attr_df)
#         # clustering_data = dict_data(clustering, shp_path, 'clustering')
#         # attr_df.at['clustering*', area] = clustering_data
#
#         clustering_weighted = nx.clustering(new_diG, weight='time')
#         attr_df = dict_data(clustering_weighted, shp_path, 'clustering_w*', area, attr_df)
#         # clustering_weighted_data = dict_data(clustering_weighted, shp_path, 'clustering_w')
#         # attr_df.at['clustering_w*', area] = clustering_weighted_data
#
#         # save_attr_df(attr_df, study_area_dir)
#
#     # ----------------------------------------------------------------------
#     # NETWORK SHAPE ATTRIBUTES
#     # 10. ECCENTRICITY: maximum distance from v to all other nodes in G
#     # Radius/Diameter of graph: radius is minimum eccentricity/The diameter is the maximum eccentricity.
#     if pd.isnull(attr_df.loc['eccentricity', area]):
#         eccentricity = nx.algorithms.distance_measures.eccentricity(new_G)
#         attr_df = dict_data(eccentricity, shp_path, 'eccentricity', area, attr_df)
#         attr_df = dict_data(min(eccentricity.values()), shp_path, 'radius', area, attr_df)
#         attr_df = dict_data(max(eccentricity.values()), shp_path, 'diameter', area, attr_df)
#         # eccentricity_data = dict_data(eccentricity, shp_path, 'eccentricity')
#         # attr_df.at['eccentricity', area] = eccentricity_data
#         # attr_df.at['radius', area] = nx.radius(new_G)
#         # attr_df.at['diameter', area] = nx.diameter(new_G)
#         # save_attr_df(attr_df, study_area_dir)
#
#     # Center: center is the set of nodes with eccentricity equal to radius.
#     # if pd.isnull(attr_df.loc['center_nodes', area]):
#     #     center = nx.algorithms.distance_measures.center(new_G)
#     #     attr_df.at['center_nodes', area] = center
#     #     print(datetime.datetime.now(), 'Center nodes: ' + str(center))
#     #     save_attr_df(attr_df, study_area_dir)
#     # # Periphery: set of nodes with eccentricity equal to the diameter.
#     # if pd.isnull(attr_df.loc['periphery_nodes', area]):
#     #     periphery = nx.algorithms.distance_measures.periphery(new_G)
#     #     attr_df.at['periphery_nodes', area] = periphery
#     #     print(datetime.datetime.now(), 'Periphery nodes: ' + str(periphery))
#     #     save_attr_df(attr_df, study_area_dir)
#     # # Baryecnter: subgraph that minimizes the function sum(w(u,v))
#     # if pd.isnull(attr_df.loc['barycenter_nodes', area]):
#     #     barycenter = nx.algorithms.distance_measures.barycenter(new_G, weight='time')
#     #     attr_df.at['barycenter_nodes', area] = barycenter
#     #     print(datetime.datetime.now(), 'Baryenter nodes: ' + str(barycenter))
#     #     save_attr_df(attr_df, study_area_dir)
#
#     # 'network_distance'
#     if pd.isnull(attr_df.loc['network_distance', area]):
#         nodes_list = new_G.edges.data('length', default=1)
#         total_len = 0
#         for nod in nodes_list:
#             total_len += nod[2]
#
#         attr_df = dict_data(total_len, shp_path, 'network_distance', area, attr_df)
#         # attr_df.at['network_distance', area] = total_len
#         # print(datetime.datetime.now(), 'Total network distance: ' + str(total_len))
#         # save_attr_df(attr_df, study_area_dir)
#
#     # ----------------------------------------------------------------------
#     # 11. OSMnx stats module
#     # if not os.path.isfile(str(shp_path) + '/' + 'stats_basic.pkl'):
#         # print(datetime.datetime.now(), 'Calculating basic_stats')
#         # new_G.graph['crs'] = 'epsg:2056'
#         # new_G.graph['name'] = str(area) + '_MultiDiGraph'
#         # basic_stats = ox.basic_stats(new_G, area=study_area_shp.area, clean_intersects=True, tolerance=15,
#         #                              circuity_dist='euclidean')
#         # with open(str(shp_path) + '/stats_basic.pkl', 'wb') as f:
#         #     pickle.dump(basic_stats, f, pickle.HIGHEST_PROTOCOL)
#     # if not os.path.isfile(str(study_area_dir) + '/' + 'stats_extended.pkl'):
#         # print('Calculating extended_stats')
#         # extended_stats = ox.extended_stats(new_G, connectivity=True, anc=True, ecc=True, bc=True, cc=True)
#         # with open(str(shp_path) + '/stats_extended.pkl', 'wb') as f:
#         #     pickle.dump(extended_stats, f, pickle.HIGHEST_PROTOCOL)
#
#     # transbase of attributes from stats_basic dic to attributes table:
#     basic_stats_list = {'n': 'n_nodes',
#                         'm': 'n_edges',
#                         'edge_length_total': 'network_distance',
#                         'intersection_count': 'n_intersection',
#                         'street_segments_count': 'n_street',
#                         'streets_per_node_avg': 'streets_per_node',
#                         'node_density_km': 'node_d_km',
#                         'intersection_density_km': 'intersection_d_km',
#                         'edge_density_km': 'edge_d_km',
#                         'street_density_km': 'street_d_km',
#                         'circuity_avg': 'circuity_avg'}
#     for attr in basic_stats_list:
#         if pd.isnull(attr_df.loc[basic_stats_list[attr], area]):
#             file = open(str(shp_path) + "/stats_basic.pkl", 'rb')
#             stats_basic_dict = pickle.load(file)
#             attr_df = dict_data(stats_basic_dict[attr], shp_path, basic_stats_list[attr], area, attr_df)
#             # attr_df.at[basic_stats_list[attr], area] = stats_basic_dict[attr]
#             # print(datetime.datetime.now(), 'Attribute ' + str(basic_stats_list[attr]) + ': '
#             #       + str(stats_basic_dict[attr]))
#             # save_attr_df(attr_df, study_area_dir)
#     print('----------------------------------------------------------------------')
#     print(datetime.datetime.now(), 'Attributes stored and saved in attr_df successfully.')
#     print('----------------------------------------------------------------------')
#
#     return attr_df, attributes


# study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
# topology_attributes(study_area_dir)
    # ideas for attributes:
    # concentration of nodes/edges in different grids
    # avg trip distance
    # distance of cg from home cluster to work/education cluster
    # take formulas from documentation (reference)
