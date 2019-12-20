import pandas as pd
import numpy as np
import pickle
import geopandas as gpd
import shapely.geometry as geo
import networkx as nx
import copy
import os
import osmnx as ox
from shapely.geometry import Point, Polygon
import datetime
from shapely.geometry.polygon import LinearRing
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
    return str([len_val, min_val,max_val,avg_val])

def cut_graph(G,shp_path, graphtype, area, study_area_shp):
    new_G = copy.deepcopy(G)
    for node in list(new_G.nodes):
        point = Point(G.nodes[node]['x'], G.nodes[node]['y'])
        in_shp = study_area_shp.contains(point)
        if not in_shp:
            new_G.remove_node(node)
    filename = str(area) + '_' + str(graphtype)
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
        attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
        print('Attr_df already exists, file loaded')
    else:
        attr_df = pd.DataFrame(data=None, columns=['area', 'n_nodes', 'n_edges', 'avg_degree', 'degree_centrality',
                                                   'avg_degree_connectivity', 'avg_edge_density',
                                                   'avg_shortest_path_duration', 'node_betweenness*',
                                                   'edge_betweenness','node_closeness', 'node_load_centrality', 'edge_load_centrality',
                                                   'clustering*', 'eccentricity', 'radius', 'diameter', 'center_nodes',
                                                   'periphery_nodes', 'barycenter_nodes'])
        print('Attributes_table does not exist, attr_df created empty')

    # Check if .csv with attributes exists:
    if os.path.isfile(str(study_area_dir) + '/' + 'stats_table.csv') == True:
        stats_df = pd.read_csv(str(study_area_dir) + '/' + 'stats_table.csv', sep=",", index_col='stats',
                              dtype=object)
        print('Stats_df already exists, file loaded')
    else:
        stats_df = pd.DataFrame(data=None,)
        print('Stats_table does not exist, stats_df created empty')
    print('----------------------------------------------------------------------')

    study_area_list = list(os.walk(study_area_dir))[0][1]
    for area in study_area_list:
        print(area, datetime.datetime.now().time())
        shp_path = str(study_area_dir) + "/" + area
        study_area_shp = gpd.read_file(str(shp_path) + "/" + area + ".shp").iloc[0]['geometry']

        # Create a buffer area for the shp file to avoid problems at the moment of calculating attributes
        # and substitute the original shp
        # buffer = 2000 #in meters
        # poly_line = LinearRing(study_area_shp.exterior)
        # poly_line_offset = poly_line.buffer(buffer, resolution=16, join_style=2, mitre_limit=1)
        # study_area_shp = Polygon(poly_line_offset.exterior)
        # how to combine polygons from a gdf and save to file the final polygon
        # m_ggdf = pd.DataFrame(data=None, columns=['id', 'geometry'])
        # new_row = {'id': 0,
        #            'geometry': study_area_shp
        #            }
        # m_ggdf = m_ggdf.append(new_row, ignore_index=True)
        # area_gdf = gpd.GeoDataFrame(m_ggdf)
        # area_gdf.to_file(str(shp_path) + "/" + area + ".shp")

        # Check if this area was filtered already by checking existance of done.txt
        if os.path.isfile(str(shp_path) + "/" + area + "_MultiDiGraph_largest.gpickle") == True:
            print('Graph already exists')
            attr_df = topology_attributes(study_area_dir, area, attr_df, stats_df, study_area_shp, nodes_dict)
            continue

        new_G = cut_graph(G, shp_path, 'MultiDiGraph', area, study_area_shp)
        new_diG = cut_graph(diG, shp_path, 'DiGraph', area, study_area_shp)

        # Create shp file with final graph
        df = pd.DataFrame(data=None, columns=['way_id', 'start_node_id', 'end_node_id', 'geometry'])
        for start, end, dup_way in list(new_G.edges):
            new_row = {'way_id': new_G[start][end][dup_way]['way_id'],
                       'start_node_id': start,
                       'end_node_id': end,
                       'geometry': geo.LineString([nodes_dict[str(start)], nodes_dict[str(end)]])
                       }
            df = df.append(new_row, ignore_index=True)
        gdf = gpd.GeoDataFrame(df)
        gdf.to_file(str(shp_path) + "/" + str(area) + "_network" + ".shp")

        # Calculate attributes for new areas that has now the graph filtered
        attr_df = topology_attributes(study_area_dir, area, attr_df, stats_df, study_area_shp, nodes_dict)
        print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('Process finished correctly: shp and graph files created in destination')

# NETWORK ANALYSIS ATTRIBUTES
def topology_attributes(study_area_dir, area, attr_df, stats_df, study_area_shp, nodes_dict):
    # graph_file = r'C:\Users\Ion\TFM\data\network_graphs\test'
    #     # study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
    #     # area = 'bern'
    new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
    new_diG = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_DiGraph_largest.gpickle')
    shp_path = str(study_area_dir) + '/' + area

    # check if table is up to date with all attributes:
    attributes = ['n_nodes',
               'n_edges',
                'area',
               'avg_degree',
                  'avg_neighbor_degree',
               'degree_centrality',
               'avg_degree_connectivity',
                  'node_connectivity',
                  'edge_connectivity',
                  'avg_node_connectivity',
               'avg_edge_density',
               'avg_shortest_path_duration',
               'node_betweenness*',
               'edge_betweenness',
                'node_straightness',
               'node_closeness*',
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

    for attribute in attributes:
        if attribute not in attr_df.index:
            s = pd.Series(name=attribute)
            attr_df = attr_df.append(s)
            print('Added ' +str(attribute) + ' as row to attr_df')

    # create empty row with areas name to add attributes
    if area not in attr_df.columns:
        new_column = pd.DataFrame({area: [len(list(new_G)), len(new_G.edges()), study_area_shp]}, index=['n_nodes', 'n_edges', 'area'])
        attr_df = pd.concat([attr_df, new_column], axis=1, sort=False)
        attr_df = attr_df.astype({area: object})
        # attr_df[area].dtype
        print('Added ' + str(area) + ' as column to attr_df as dtype: ' + str(attr_df[area].dtype))
    else:
        print('Area exists in attributes table, checking for additional attributes')

    # ----------------------------------------------------------------------
    # 1. AVG DEGREE OF GRAPH: number of edges adjacent per node
    if pd.isnull(attr_df.loc['avg_degree', area]) == True:
        # attr_df.at['avg_degree', area]
        degree = nx.degree(new_G)
        degree_list = list(degree)
        sum_deg = 0
        count_deg = 0
        for node, deg in degree_list:
            sum_deg += deg
            count_deg += 1
        avg_deg = sum_deg/count_deg
        attr_df.at['avg_degree', area] = avg_deg
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
        print('Average degree(edges adj per node): ' + str(avg_deg))
    if pd.isnull(attr_df.loc['avg_neighbor_degree', area]) == True:
        avg_neighbor_degree = nx.average_neighbor_degree(new_G)
        avg_neighbor_degree_data = dict_data(avg_neighbor_degree, shp_path, 'avg_neighbor_degree')
        attr_df.at['avg_neighbor_degree', area] = avg_neighbor_degree_data
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # ----------------------------------------------------------------------
    # 2. DEGREE OF CENTRALITY:
    if pd.isnull(attr_df.loc['degree_centrality', area]) == True:
        degree_centr = nx.algorithms.degree_centrality(new_G)
        degree_centr_data = dict_data(degree_centr, shp_path, 'degree_centrality')
        attr_df.at['degree_centrality', area] = degree_centr_data
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # ----------------------------------------------------------------------
    # 3. CONNECTIVITY: For a node of degree k - What is the average of its neighbours' degree?
    if pd.isnull(attr_df.loc['avg_degree_connectivity', area]) == True:
        avg_degree_connect = nx.average_degree_connectivity(new_G)
        avg_degree_connect_data = dict_data(avg_degree_connect, shp_path, 'avg_degree_connect')
        attr_df.at['avg_degree_connectivity', area] = avg_degree_connect_data
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    if pd.isnull(attr_df.loc['node_connectivity', area]) == True:
        node_connect = nx.node_connectivity(new_G)
        attr_df.at['node_connectivity', area] = node_connect
        print('Node connectivity k: ' + str(node_connect))
        edge_connect = nx.edge_connectivity(new_G)
        attr_df.at['edge_connectivity', area] = edge_connect
        print('Node connectivity k: ' + str(edge_connect))
        avg_node_connect =nx.average_node_connectivity(new_G)
        attr_df.at['avg_node_connectivity', area] = avg_node_connect
        print('Node connectivity k: ' + str(avg_node_connect))
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # ----------------------------------------------------------------------
    # 4. AVG EDGE DENSITY: Average edge density of the Graphs
    if pd.isnull(attr_df.loc['avg_edge_density', area]) == True:
        avg_edge_density = nx.density(new_G)
        print('Average edge density: ' + str(avg_edge_density))
        attr_df.at['avg_edge_density', area] = avg_edge_density
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # ----------------------------------------------------------------------
    # 5. AVG SHORTEST PATH LENGTH (WEIGHTED BY TIME):
    if pd.isnull(attr_df.loc['avg_shortest_path_duration', area]) == True:
        avg_spl = nx.average_shortest_path_length(new_G, weight='time')
        print('Average shortest path length (duration in seconds): ' + str(avg_spl))
        attr_df.at['avg_shortest_path_duration', area] = avg_spl
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # ----------------------------------------------------------------------
    # 6. BETWEENNESS CENTRALITY: how many times a node or edge is passed for the sp
    # nodes betweenness
    if pd.isnull(attr_df.loc['node_betweenness*', area]) == True:
        node_betw_centr = nx.betweenness_centrality(new_diG, weight='time')
        node_betw_centr_data = dict_data(node_betw_centr, shp_path, 'node_betweenness')
        attr_df.at['node_betweenness*', area] = node_betw_centr_data
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # edges betweenness
    if pd.isnull(attr_df.loc['edge_betweenness', area]) == True:
        edge_betw_centr = nx.edge_betweenness_centrality(new_G, weight='time')
        edge_betw_centr_data = dict_data(edge_betw_centr, shp_path, 'edge_betweenness')
        attr_df.at['edge_betweenness', area] = edge_betw_centr_data
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # ----------------------------------------------------------------------
    # 7. CLOSENESS CENTRALITY: Of a node is the average length of the shortest path from the node to all other nodes
    # nodes betweenness
    if pd.isnull(attr_df.loc['node_closeness*', area]) == True:
        node_close_centr = nx.betweenness_centrality(new_diG, weight='time')
        node_close_centr_data = dict_data(node_close_centr, shp_path, 'node_closeness')
        attr_df.at['node_closeness*', area] = node_close_centr_data
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])

    # ----------------------------------------------------------------------
    # 8. STRAIGHTNESS CENTRALITY: compares the shortest path with the euclidean distance of each pair of nodes
    if pd.isnull(attr_df.loc['node_straightness', area]) == True:
        node_straightness = {}
        for i in list(new_G.nodes):
            i_lonlat = Point(nodes_dict[str(i)])
            dist_comp_list = []
            for j in list(new_G.nodes):
                if i == j:
                    continue
                j_lonlat = Point(nodes_dict[str(j)])
                eucl_dist = i_lonlat.distance(j_lonlat)
                sp_dist = nx.dijkstra_path_length(new_G, i, j, weight='length')
                dist_comp = eucl_dist / sp_dist

                dist_comp_list.append(dist_comp)
            straightness = (1/len(list(new_G.nodes))-1)*sum(dist_comp_list)
            node_straightness[i] = straightness

        node_straightness_data = dict_data(node_straightness, shp_path, 'node_straightness')
        attr_df.at['node_straightness', area] = node_straightness_data
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])

    # ----------------------------------------------------------------------
    # 8. LOAD CENTRALITY:counts the number of shortest paths which cross each node/edge
    # nodes load:
    if pd.isnull(attr_df.loc['node_load_centrality', area]) == True:
        load_centrality = nx.load_centrality(new_G)
        load_centrality_data = dict_data(load_centrality, shp_path, 'node_load_centrality')
        attr_df.at['node_load_centrality', area] = load_centrality_data
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # edges load:
    if pd.isnull(attr_df.loc['edge_load_centrality', area]) == True:
        edge_load = nx.edge_load_centrality(new_G)
        edge_load_data = dict_data(edge_load, shp_path, 'edge_load_centrality')
        attr_df.at['edge_load_centrality', area] = edge_load_data
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])

    # ----------------------------------------------------------------------
    # 9. CLUSTERING: geometric average of the subgraph edge weights
    if pd.isnull(attr_df.loc['clustering*', area]) == True:
        clustering = nx.clustering(new_diG)
        clustering_data = dict_data(clustering, shp_path, 'clustering')
        attr_df.at['clustering*', area] = clustering_data

        clustering_weighted = nx.clustering(new_diG, weight='time')
        clustering_weighted_data = dict_data(clustering_weighted, shp_path, 'clustering')
        attr_df.at['clustering_w*', area] = clustering_weighted_data

        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])

    # ----------------------------------------------------------------------
    # NETWORK SHAPE ATTRIBUTES
    # 10. EXCENTRICITY: maximum distance from v to all other nodes in G
    # Radius/Diameter of graph: radius is minimum eccentricity/The diameter is the maximum eccentricity.
    if pd.isnull(attr_df.loc['eccentricity', area]) == True:
        eccentricity = nx.algorithms.distance_measures.eccentricity(new_G)
        eccentricity_data = dict_data(eccentricity, shp_path, 'eccentricity')
        attr_df.at['eccentricity', area] = eccentricity_data
        attr_df.at['radius', area] = nx.radius(new_G)
        attr_df.at['diameter', area] = nx.diameter(new_G)
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])

    # Center: center is the set of nodes with eccentricity equal to radius.
    if pd.isnull(attr_df.loc['center_nodes', area]) == True:
        center = nx.algorithms.distance_measures.center(new_G)
        attr_df.at['center_nodes', area] = center
        print('Center nodes: ' + str(center))
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # Periphery: set of nodes with eccentricity equal to the diameter.
    if pd.isnull(attr_df.loc['periphery_nodes', area]) == True:
        periphery = nx.algorithms.distance_measures.periphery(new_G)
        attr_df.at['periphery_nodes', area] = periphery
        print('Periphery nodes: ' + str(periphery))
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # Baryecnter: subgraph that minimizes the function sum(w(u,v))
    if pd.isnull(attr_df.loc['barycenter_nodes', area]) == True:
        barycenter = nx.algorithms.distance_measures.barycenter(new_G, weight='time')
        attr_df.at['barycenter_nodes', area] = barycenter
        print('Baryenter nodes: ' + str(barycenter))
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])

    # ----------------------------------------------------------------------
    # 11. OSMnx stats module
    if os.path.isfile(str(study_area_dir) + '/' + 'stats_basic.pkl') == False:
        print('Calculating basic_stats')
        new_G.graph['crs'] = 'epsg:2056'
        new_G.graph['name'] = str(area) + '_MultiDiGraph'
        basic_stats = ox.basic_stats(new_G, area=study_area_shp.area, clean_intersects=True,tolerance=15, circuity_dist='euclidean')
        with open(str(shp_path) + '/stats_basic.pkl', 'wb') as f:
            pickle.dump(basic_stats, f, pickle.HIGHEST_PROTOCOL)
    # if os.path.isfile(str(study_area_dir) + '/' + 'stats_extended.pkl') == False:
        # print('Calculating extended_stats')
        # extended_stats = ox.extended_stats(new_G, connectivity=True, anc=True, ecc=True, bc=True, cc=True)
        # with open(str(shp_path) + '/stats_extended.pkl', 'wb') as f:
        #     pickle.dump(extended_stats, f, pickle.HIGHEST_PROTOCOL)

    # Create shp file with final graph
    # attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    # stats_df.to_csv(str(study_area_dir) + "/stats_table.csv", sep=",", index=True, index_label=['stats'])
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
