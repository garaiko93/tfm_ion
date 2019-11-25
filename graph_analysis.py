import pandas as pd
import pickle
import geopandas as gpd
import shapely.geometry as geo
import networkx as nx
import copy
import os
from shapely.geometry import Point

from network_graph import check_iso_graph

# Filter graph and create a new one with nodes only into the shp
def filter_graph(study_area_dir, graph_file):
    print('Filtering graph process starts')
    # IMPORT data: graph and nodes_dict:
    G = nx.read_gpickle(str(graph_file)+'\ch_network_largest_graph_bytime.gpickle')
    file = open(str(graph_file) + "\ch_nodes_dict2056.pkl", 'rb')
    nodes_dict = pickle.load(file)

    study_area_list = list(os.walk(study_area_dir))[0][1]
    print('----------------------------------------------------------------------')
    for area in study_area_list:
        print(area)
        new_G = copy.deepcopy(G)
        shp_path = str(study_area_dir) + "/" + area
        study_area_shp = gpd.read_file(str(shp_path) + "/" + area + ".shp").iloc[0]['geometry']
        for node in list(new_G.nodes):
            point = Point(G.nodes[node]['x'], G.nodes[node]['y'])
            in_shp = study_area_shp.contains(point)
            if not in_shp:
                new_G.remove_node(node)
            elif in_shp:
                pass
            else:
                print(node,[point.x,point.y])

        [new_G, isolated, largest] = check_iso_graph(new_G, shp_path)
        #     Create shp file with final graph
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
        print('Process finished correctly: shp and graph file created in destination')

        # NETWORK ANALYSIS ATTRIBUTES
        # nx.algorithms.degree_centrality(new_G)
        # nx.density(new_G)  # Average edge density of the Graphs
        # nx.average_shortest_path_length(new_G, weight='time')  # Average shortest path length for ALL paths in the Graph
        # nx.average_degree_connectivity(new_G) # For a node of degree k - What is the average of its neighbours' degree?
        print('----------------------------------------------------------------------')