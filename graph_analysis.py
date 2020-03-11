import pandas as pd
import pickle
import geopandas as gpd
import shapely.geometry as geo
import networkx as nx
import igraph as ig
import copy
import os
import math
import ast
# import osmnx as ox
from scipy import spatial
from progressbar import Percentage, ProgressBar, Bar, ETA
from shapely.geometry import Point, Polygon, LinearRing
import datetime
import ntpath

# functions from other scripts
from network_graph import check_iso_graph
from population_distribution import create_distrib


def save_attr_df(attr_df, study_area_dir, attr_avg_df=None, attr_avg_dfT=None):
    # save = False        # uncomment to NOT save (testing purposes)
    save = True         # uncomment to save
    if save:
        attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])

        if attr_avg_df is not None:
            attr_avg_df.to_csv(str(study_area_dir) + "/attribute_table_AVG.csv", sep=",", index=True,
                               index_label=['attributes'])
        if attr_avg_dfT is not None:
            attr_avg_dfT.to_csv(str(study_area_dir) + "/attribute_table_AVG_T.csv", sep=",", index=True,
                                index_label=['study_area'])
        # print(datetime.datetime.now(), 'Dataframes saved: attribute_table, attribute_table_AVG and '
        #                                'attribute_table_AVG_T were created and saved.')
    else:
        print(datetime.datetime.now(), 'For test purposes attribute dataframes were not saved.')

def create_igraph(new_G):
    # create igraph with networkx graph info
    print(datetime.datetime.now(), 'Creating graph with igraph ...')
    g = ig.Graph(directed=True)
    for node in new_G.nodes():
        g.add_vertex(name=str(node))
    for edge in new_G.edges.data():
        g.add_edge(str(edge[0]), str(edge[1]),
                   time=edge[2]['time'],
                   name=str(edge[2]['way_id']),
                   modes=edge[2]['modes'],
                   length=edge[2]['length'])
    return g


def btw_acc(new_G, chG, area_path, area, nodes_dict, attr_df, grid_size=500):
    # import nodes into study area: new_G.nodes()
    # import graph of full ch and transform into igraph
    # iterate over all pair of nodes in the study areas nodes by a maximum time
    # file = open(r'C:/Users/Ion/TFM/data/network_graphs/ch_nodes_dict2056.pkl', 'rb')
    # nodes_dict = pickle.load(file)
    # chG = nx.read_gpickle(r'C:/Users/Ion/TFM/data/study_areas/test_area/test_area_MultiDiGraph5k_largest.gpickle')
    # new_G = nx.read_gpickle(r'C:/Users/Ion/TFM/data/study_areas/test_area/test_area_MultiDiGraph_largest.gpickle')
    # area_path = r'C:/Users/Ion/TFM/data/study_areas/test_area'
    # area = 'test_area'

    print(datetime.datetime.now(), 'Calculating lim_edge_betweenness of graph ...')
    time_lim = 1200 #seconds
    grid_size = 300 #meters
    g = create_igraph(chG)
    # g = create_igraph(new_G)

    # call function to create df with grid areas defined in df with population, employment and optional facility values of study area
    m_df = create_distrib(area_path, grid_size, False)

    # find closest node of centroid:
    # Build tree for KDTree nearest neighbours search, in G only start and end nodes are included
    G_nodes = list(new_G.nodes)
    G_lonlat = []
    for node in G_nodes:
        lonlat = nodes_dict[str(node)]
        G_lonlat.append(lonlat)
    print(datetime.datetime.now(), 'KDTree has: ' + str(len(G_lonlat)) + ' nodes.')

    tree = spatial.KDTree(G_lonlat)
    for index, row in m_df.iterrows():
        # id = row['id']
        centroid = Point(row['centroid'][0], row['centroid'][1])

        nn = tree.query(centroid)
        coord = G_lonlat[nn[1]]
        closest_node_id = int(list(nodes_dict.keys())[list(nodes_dict.values()).index((coord[0], coord[1]))])
        m_df.at[index, 'closest_node'] = int(closest_node_id)
    print(datetime.datetime.now(), 'Closest node of every grid found.')

    # add columns of ACC empl and ACC pop respectively to the df and export as shapefile:
    # for facil in ['pop', 'empl', 'opt']:
    for facil in ['pop', 'empl']:
    # for facil in ['pop']:
        print(datetime.datetime.now(), 'Calculating ' + str(facil) + ' accessibility for each grid area ...')
        for index, row in m_df.iterrows():
            j = str(int(row['closest_node']))
            facil_j = row[facil]  # iterate over pop, empl, opt
            acc_list = []
            for index2, row2 in m_df.iterrows():
                k = str(int(row2['closest_node']))
                if j == k:
                    continue
                # find sp and sum time
                paths = g.get_shortest_paths(v=j, to=k, weights='time', mode='OUT', output="epath")
                path_time = 0
                for i in paths[0]:
                    path_time += g.es[i]['time']

                # define impedance function
                acc = facil_j * math.exp(-0.05*path_time)
                acc_list.append(acc)
            m_df.at[index, 'acc_' + str(facil)] = sum(acc_list)
        print(datetime.datetime.now(), 'Computation of ' + str(facil) + ' accessibility finished.')

    m_gdf = gpd.GeoDataFrame(m_df.loc[:, m_df.columns != 'centroid'])
    m_gdf.to_file(str(area_path) + "/" + str(area) + '_' + "facility_distribution_" +
                  str(grid_size) + "gs.shp")

    # Compute btw_accessibility metrics:
        # First metric ('cPop_li'): quantifies the potential level of a link’s exposure in terms of trip
    # production (population) and attraction
        # Second metric ('cApop_li'): quantifies the importance of elements in a network with
    # respect to their contributions to the generation of accessibility. This indicator
    # focuses on the potential to reach opportunities through each element of the
    # network
    def metrics_comput(m_df, facil, other_facil):
        # facil = 'pop'
        # other_facil = 'empl'
        cPop_li = {}
        cApop_li = {}
        print(datetime.datetime.now(), 'Calculating ' + str(facil) + ' betweenness-accessibility metric ...')
        for index, row_j in m_df.iterrows():
            j = str(int(row_j['closest_node']))
            acc_j = row_j['acc_' + str(other_facil)]  # iterate over pop, empl
            facil_j = row_j[facil]
            if acc_j == 0:
                continue

            for index2, row_k in m_df.iterrows():
                k = str(int(row_k['closest_node']))
                facil_k = row_k[other_facil]
                facil2_k = row_k[facil]
                if j == k:
                    continue

                # Compute shortest paths for j, k
                paths = g.get_shortest_paths(v=j, to=k, weights='time', mode='OUT', output="epath")
                path_time = 0
                way_sp = {} #stores the number of occurrencies of each link in the shortest paths: d[way_id] = int
                for path in paths:
                    for i in path:
                        path_time += g.es[i]['time']
                        way_id = g.es[i]['name']
                        if not way_sp.get(way_id):
                            way_sp[way_id] = 1
                        else:
                            way_sp[way_id] += 1

                # Compute weight and centrality value for every link in paths
                f_tt = math.exp(-0.05 * path_time)  # define impedance function
                weight_C_facil = facil_j * ((facil_k * f_tt)/acc_j)
                weight_Ca_facil = facil2_k * f_tt

                for link in list(way_sp):
                    c_pop = (way_sp[link] / len(paths)) * weight_C_facil
                    cA_pop = (way_sp[link] / len(paths)) * weight_Ca_facil
                    if not cPop_li.get(link):
                        cPop_li[link] = [c_pop]
                        cApop_li[link] = [cA_pop]
                    else:
                        a_list = cPop_li[link]
                        a_list.append(c_pop)
                        cPop_li[link] = a_list

                        b_list = cApop_li[link]
                        b_list.append(cA_pop)
                        cApop_li[link] = b_list

        # Finally, for every link, compute sum of every pair of nodes, obtaining betweenness values
        # Define absolute values for normalization:
        tot_pop = m_df[facil].sum()
        tot_acc = m_df['acc_' + str(facil)].sum()

        # Load geodataframe with every link of network to add btw-acc values
        if os.path.isfile(str(area_path) + "/" + str(area) + "_btw_acc.shp"):
            links_gdf = gpd.read_file(str(area_path) + "/" + str(area) + "_btw_acc.shp")
        else:
            links_gdf = gpd.read_file(str(area_path) + "/" + str(area) + "_network_5k.shp")
            links_gdf['btw_check'] = 0

        links_gdf['btw_' + str(facil)] = 0
        links_gdf['btw_Acc_' + str(facil)] = 0
        for link in list(cPop_li):
            cPop_li[link] = sum(cPop_li[link])/tot_pop
            cApop_li[link] = sum(cApop_li[link])/tot_acc

            # update gdf of network with obtained values
            ix = links_gdf.index[links_gdf['way_id'] == link][0]

            links_gdf.loc[ix, 'btw_' + str(facil)] = cPop_li[link]
            links_gdf.loc[ix, 'btw_Acc_' + str(facil)] = cApop_li[link]
            links_gdf.loc[ix, 'btw_check'] = 1

        links_gdf.to_file(str(area_path) + "/" + str(area) + "_btw_acc.shp")
        return cPop_li, cApop_li, links_gdf

    cPop_li, cApop_li, links_gdf = metrics_comput(m_df=m_df, facil='pop', other_facil='empl')
    cEmpl_li, cAempl_li, links_gdf = metrics_comput(m_df=m_df, facil='empl', other_facil='pop')

    # Clean links which did not take part into any shortest path of metric computation
    indexes = []
    for index3, row3 in links_gdf.iterrows():
        btw_check = row3['btw_check']
        if btw_check != 1:
            indexes.append(index3)
    links_gdf = links_gdf.drop(links_gdf.index[indexes])
    links_gdf.to_file(str(area_path) + "/" + str(area) + "_btw_acc.shp")

    # Export dictionaries and save values in attributes table
    dict_data(cPop_li, area_path, 'btw_home_trip_production', area, attr_df)
    dict_data(cEmpl_li, area_path, 'btw_empl_trip_generation', area, attr_df)

    dict_data(cApop_li, area_path, 'btw_acc_trip_generation', area, attr_df)
    dict_data(cAempl_li, area_path, 'btw_acc_trip_production', area, attr_df)

    print(datetime.datetime.now(), 'Betweenness-accessibility metrics computation finished.')
    # return edge_btw_acc

def straightness(nodes_dict, new_G, nodes_list):
    print(datetime.datetime.now(), 'Calculating node_straightness of graph ...')
    print(len(nodes_list))
    g = create_igraph(new_G)
    n = 0
    node_straightness = {}
    # for i in g.vs['name']:
    for i in nodes_list:
        # print(datetime.datetime.now(), n)
        if n % 1 == 0:
            print(datetime.datetime.now(), n)
        i_lonlat = Point(nodes_dict[str(i)])
        dist_comp_list = []
        for j in g.vs['name']:
            if i == j:
                continue
            sp_dist = 0
            j_lonlat = Point(nodes_dict[str(j)])
            eucl_dist = i_lonlat.distance(j_lonlat)
            path = g.get_shortest_paths(v=i, to=j, weights='time', mode='OUT', output="epath")
            for k in path[0]:
                sp_dist += g.es[k]['length']
            dist_comp = eucl_dist / sp_dist
            dist_comp_list.append(dist_comp)
        node_straightness[i] = (1 / (len(g.vs) - 1)) * sum(dist_comp_list)
        n += 1
    return node_straightness


def dict_data(dicti, shp_path, attrib_name, area, attr_df):
    # if '*' in attrib_name: #this means DiGraph was used to compute attribute instead of MultiDiGraph
    filename = ntpath.split(attrib_name)[1].split('*')[0]
    study_area_dir = ntpath.split(shp_path)[0]
    if isinstance(dicti, dict):
        len_val = len(dicti)
        min_val = min(dicti.values())
        max_val = max(dicti.values())
        avg_val = sum(dicti.values()) / len(dicti)
        value = str([len_val, min_val, max_val, avg_val])
        with open(str(shp_path) + '/attr_' + str(filename) + '.pkl', 'wb') as f:
            pickle.dump(dicti, f, pickle.HIGHEST_PROTOCOL)
    else:
        value = dicti

    attr_df.at[attrib_name, area] = value
    save_attr_df(attr_df, study_area_dir)
    print(datetime.datetime.now(), 'Attribute ' + str(attrib_name) + ' calculated: ' + str(value))
    return attr_df

def cut_graph(G, shp_path, graphtype, area, study_area_shp):
    new_G = copy.deepcopy(G)
    for node in list(new_G.nodes):
        point = Point(G.nodes[node]['x'], G.nodes[node]['y'])
        in_shp = study_area_shp.contains(point)
        if not in_shp:
            new_G.remove_node(node)
    filename = str(area) + '_' + str(graphtype)
    [new_G, isolated, largest] = check_iso_graph(new_G, shp_path, filename)
    return new_G


# Filter graph and create a new one with nodes only into the shp
def filter_graph(study_area_dir, graph_file, area_def=None):
    print(datetime.datetime.now(), 'Filtering graph process starts')
    # IMPORT data: graph and nodes_dict:
    G = nx.read_gpickle(str(graph_file)+'/ch_MultiDiGraph_bytime_largest.gpickle')
    diG = nx.read_gpickle(str(graph_file) + '/ch_DiGraph_bytime_largest.gpickle')
    file = open(str(graph_file) + "/ch_nodes_dict2056.pkl", 'rb')
    nodes_dict = pickle.load(file)
    print('----------------------------------------------------------------------')

    # Check if .csv with attributes exists:
    if os.path.isfile(str(study_area_dir) + '/' + 'attribute_table.csv'):
        attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
        print(datetime.datetime.now(), 'Attr_df already exists, file loaded')
    else:
        attr_df = pd.DataFrame(data=None)
        print(datetime.datetime.now(), 'Attributes_table does not exist, attr_df created empty')

    # Check if .csv with attributes exists:
    # if os.path.isfile(str(study_area_dir) + '/' + 'stats_table.csv'):
    #     stats_df = pd.read_csv(str(study_area_dir) + '/' + 'stats_table.csv', sep=",", index_col='stats',
    #                           dtype=object)
    #     print(datetime.datetime.now(), 'Stats_df already exists, file loaded')
    # else:
    #     stats_df = pd.DataFrame(data=None,)
    #     print(datetime.datetime.now(), 'Stats_table does not exist, stats_df created empty')
    print('----------------------------------------------------------------------')
    # if area:
    #     shp_path = str(study_area_dir) + "/" + area
    #     study_area_shp = gpd.read_file(str(shp_path) + "/" + area + ".shp").iloc[0]['geometry']
    #     if os.path.isfile(str(shp_path) + "/" + area + "_MultiDiGraph_largest.gpickle"):
    #         print(datetime.datetime.now(), 'Graph already exists')
    #         attr_df, attributes = topology_attributes(study_area_dir, area, attr_df, study_area_shp, nodes_dict, G)
    # else:
    study_area_list = list(os.walk(study_area_dir))[0][1]
    for area in study_area_list:
        if area_def:
            area = area_def
        print(datetime.datetime.now(), area)
        shp_path = str(study_area_dir) + "/" + str(area)
        study_area_shp = gpd.read_file(str(shp_path) + "/" + str(area) + ".shp").iloc[0]['geometry']

        # Check if this area was filtered already by checking existance of done.txt
        if os.path.isfile(str(shp_path) + "/" + str(area) + "_network_5k.shp"):
            print(datetime.datetime.now(), 'Graph already exists')
            attr_df, attributes = topology_attributes(study_area_dir, area, attr_df, study_area_shp, nodes_dict)
            # Manipulate attributes table
            take_avg(attr_df, study_area_dir, attributes)
            if area_def:
                break
            else:
                continue
        print(datetime.datetime.now(), 'Creating graph for area ...')
        new_G = cut_graph(G, shp_path, 'MultiDiGraph', area, study_area_shp)
        new_diG = cut_graph(diG, shp_path, 'DiGraph', area, study_area_shp)

        poly_line = LinearRing(study_area_shp.exterior)
        poly_line_offset = poly_line.buffer(5000, resolution=16, join_style=2, mitre_limit=1)
        study_area_shp30k = Polygon(list(poly_line_offset.exterior.coords))
        # study_area_shp30k = Polygon(study_area_shp.buffer(5000).exterior, [study_area_shp.exterior])

        new_G30k = cut_graph(G, shp_path, 'MultiDiGraph5k', area, study_area_shp30k)
        # create new graph with 30k buffer area

        # Create shp file with final graph
        def export_shp(new_G, filename):
            print(datetime.datetime.now(), 'Creating shp file of network ...')
            df = pd.DataFrame(data=None, columns=['way_id', 'start_node_id', 'end_node_id', 'geometry'])
            for start, end, dup_way in list(new_G.edges):
                new_row = {'way_id': new_G[start][end][dup_way]['way_id'],
                           'start_node_id': start,
                           'end_node_id': end,
                           'geometry': geo.LineString([nodes_dict[str(start)], nodes_dict[str(end)]])
                           }
                df = df.append(new_row, ignore_index=True)
            gdf = gpd.GeoDataFrame(df)
            gdf.to_file(str(filename))

        # Call function to export shp file
        export_shp(new_G, str(shp_path) + "/" + str(area) + "_network.shp")
        export_shp(new_G30k, str(shp_path) + "/" + str(area) + "_network_5k.shp")
        # Calculate attributes for new areas that has now the graph filtered
        attr_df, attributes = topology_attributes(study_area_dir, area, attr_df, study_area_shp, nodes_dict)
        # Manipulate attributes table
        take_avg(attr_df, study_area_dir, attributes)
        if area_def:
            break
        print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print(datetime.datetime.now(), 'Process finished correctly: shp and graph files created in output directory')


# NETWORK ANALYSIS ATTRIBUTES
def topology_attributes(study_area_dir, area, attr_df, study_area_shp, nodes_dict):
    # graph_file = r'C:\Users\Ion\TFM\data\network_graphs\test'
    study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
    area = 'freiburg'

    new_G = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph_largest.gpickle')
    new_diG = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_DiGraph_largest.gpickle')
    new_G30k = nx.read_gpickle(str(study_area_dir) + '/' + area + '/' + area + '_MultiDiGraph5k_largest.gpickle')
    shp_path = str(study_area_dir) + '/' + area

    # check if table is up to date with all attributes:
    attributes = ['n_nodes',            # stats_basic
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
        print(datetime.datetime.now(), 'Added ' + str(area) + ' as column to attr_df as dtype: ' +
              str(attr_df[area].dtype))
    else:
        print(datetime.datetime.now(), 'Area exists in attributes table, checking for additional attributes')

    # ----------------------------------------------------------------------
    # 0.Basic information of study area
    if pd.isnull(attr_df.loc['area', area]):
        area_val = study_area_shp.area
        attr_df = dict_data(area_val, shp_path, 'area', area, attr_df)
        # attr_df.at['area', area] = area_val
        # save_attr_df(attr_df, study_area_dir)
        print(datetime.datetime.now(), 'Area of study area: ' + str(area_val))
    # ----------------------------------------------------------------------
    # 1. AVG DEGREE OF GRAPH: number of edges adjacent per node
    if pd.isnull(attr_df.loc['avg_degree', area]):
        degree = nx.degree(new_G)
        degree_list = list(degree)
        sum_deg = 0
        count_deg = 0
        for node, deg in degree_list:
            sum_deg += deg
            count_deg += 1
        avg_deg = sum_deg/count_deg
        attr_df = dict_data(avg_deg, shp_path, 'avg_degree', area, attr_df)
        # attr_df.at['avg_degree', area] = avg_deg
        # save_attr_df(attr_df, study_area_dir)
        print(datetime.datetime.now(), 'Average degree(edges adj per node): ' + str(avg_deg))
    # avg_neighbour_degree:  average degree of the neighborhood of each node
    if pd.isnull(attr_df.loc['avg_neighbor_degree', area]):
        avg_neighbor_degree = nx.average_neighbor_degree(new_G)
        attr_df = dict_data(avg_neighbor_degree, shp_path, 'avg_neighbor_degree', area, attr_df)
        # avg_neighbor_degree_data = dict_data(avg_neighbor_degree, shp_path, 'avg_neighbor_degree')
        # attr_df.at['avg_neighbor_degree', area] = avg_neighbor_degree_data
        # save_attr_df(attr_df, study_area_dir)
    # ----------------------------------------------------------------------
    # 2. DEGREE OF CENTRALITY: The degree centrality for a node v is the fraction of nodes it is connected to (normalized)
    if pd.isnull(attr_df.loc['degree_centrality', area]):
        degree_centr = nx.algorithms.degree_centrality(new_G)
        attr_df = dict_data(degree_centr, shp_path, 'degree_centrality', area, attr_df)

        # degree_centr_data = dict_data(degree_centr, shp_path, 'degree_centrality')
        # attr_df.at['degree_centrality', area] = degree_centr_data
        # save_attr_df(attr_df, study_area_dir)
    # ----------------------------------------------------------------------
    # 3. CONNECTIVITY:  is the average nearest neighbor degree of nodes with degree k
    if pd.isnull(attr_df.loc['avg_degree_connectivity', area]):
        avg_degree_connect = nx.average_degree_connectivity(new_G)
        attr_df = dict_data(avg_degree_connect, shp_path, 'avg_degree_connectivity', area, attr_df)
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
    if pd.isnull(attr_df.loc['avg_edge_density', area]):
        avg_edge_density = nx.density(new_G)
        attr_df = dict_data(avg_edge_density, shp_path, 'avg_edge_density', area, attr_df)
        # attr_df.at['avg_edge_density', area] = avg_edge_density
        # save_attr_df(attr_df, study_area_dir)
        # print(datetime.datetime.now(), 'Average edge density: ' + str(avg_edge_density))

    # ----------------------------------------------------------------------
    # 5. AVG SHORTEST PATH LENGTH (WEIGHTED BY TIME):
    if pd.isnull(attr_df.loc['avg_shortest_path_duration', area]):
        avg_spl = nx.average_shortest_path_length(new_G, weight='time')
        attr_df = dict_data(avg_spl, shp_path, 'avg_shortest_path_duration', area, attr_df)
        # attr_df.at['avg_shortest_path_duration', area] = avg_spl
        # save_attr_df(attr_df, study_area_dir)
        # print(datetime.datetime.now(), 'Average shortest path length (duration in seconds): ' + str(avg_spl))
    # ----------------------------------------------------------------------
    # 6. BETWEENNESS CENTRALITY: the fraction of nodes/edges of how many times is passed for the sp
    # nodes betweenness: not agree with the result of this algorithm based on the definition of betweenness...
    if pd.isnull(attr_df.loc['node_betweenness*', area]):
        node_betw_centr = nx.betweenness_centrality(new_diG, weight='time')
        attr_df = dict_data(node_betw_centr, shp_path, 'node_betweenness*', area, attr_df)
        # node_betw_centr_data = dict_data(node_betw_centr, shp_path, 'node_betweenness')
        # attr_df.at['node_betweenness*', area] = node_betw_centr_data
        # save_attr_df(attr_df, study_area_dir)
    # edges betweenness
    if pd.isnull(attr_df.loc['edge_betweenness', area]):
        edge_betw_centr = nx.edge_betweenness_centrality(new_G, weight='time')
        attr_df = dict_data(edge_betw_centr, shp_path, 'edge_betweenness', area, attr_df)
        # edge_betw_centr_data = dict_data(edge_betw_centr, shp_path, 'edge_betweenness')
        # attr_df.at['edge_betweenness', area] = edge_betw_centr_data
        # save_attr_df(attr_df, study_area_dir)
    # if pd.isnull(attr_df.loc['btw_acc_trip_production', area]):
    btw_acc(new_G, new_G30k, shp_path, area, nodes_dict, attr_df)
        # dict_data(lim_edge_betw_centr, shp_path, 'lim_edge_betweenness*', area, attr_df)
        # lim_edge_betw_centr_data = dict_data(lim_edge_betw_centr, shp_path, 'lim_edge_betweenness')
        # attr_df.at['lim_edge_betweenness', area] = lim_edge_betw_centr_data
        # save_attr_df(attr_df, study_area_dir)
    # ----------------------------------------------------------------------
    # 7. CLOSENESS CENTRALITY: Of a node is the average length of the shortest path from the node to all other nodes
    if pd.isnull(attr_df.loc['node_closeness_time*', area]):
        node_close_time_centr = nx.closeness_centrality(new_diG, distance='time')
        attr_df = dict_data(node_close_time_centr, shp_path, 'node_closeness_time*', area, attr_df)
        # node_close_centr_data = dict_data(node_close_time_centr, shp_path, 'node_closeness_time')
        # attr_df.at['node_closeness_time*', area] = node_close_centr_data
        # save_attr_df(attr_df, study_area_dir)
    if pd.isnull(attr_df.loc['node_closeness_length*', area]):
        node_close_dist_centr = nx.closeness_centrality(new_diG, distance='length')
        attr_df = dict_data(node_close_dist_centr, shp_path, 'node_closeness_length*', area, attr_df)
        # node_close_dist_data = dict_data(node_close_dist_centr, shp_path, 'node_closeness_length')
        # attr_df.at['node_closeness_length*', area] = node_close_dist_data
        # save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # 8. STRAIGHTNESS CENTRALITY: compares the shortest path with the euclidean distance of each pair of nodes
    # if pd.isnull(attr_df.loc['node_straightness', area]):
    #     node_straightness = straightness(nodes_dict, new_G)
    #     node_straightness_data = dict_data(node_straightness, shp_path, 'node_straightness')
    #     attr_df.at['node_straightness', area] = node_straightness_data
    #     save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # 8. LOAD CENTRALITY:counts the fraction of shortest paths which cross each node/edge
    # nodes load: of a node is the fraction of all shortest paths that pass through that node.
    if pd.isnull(attr_df.loc['node_load_centrality', area]):
        load_centrality = nx.load_centrality(new_G)
        attr_df = dict_data(load_centrality, shp_path, 'node_load_centrality', area, attr_df)
        # load_centrality_data = dict_data(load_centrality, shp_path, 'node_load_centrality')
        # attr_df.at['node_load_centrality', area] = load_centrality_data
        # save_attr_df(attr_df, study_area_dir)
    # edges load: counts the number of shortest paths which cross each edge
    if pd.isnull(attr_df.loc['edge_load_centrality', area]):
        edge_load = nx.edge_load_centrality(new_G)
        attr_df = dict_data(edge_load, shp_path, 'edge_load_centrality', area, attr_df)
        # edge_load_data = dict_data(edge_load, shp_path, 'edge_load_centrality')
        # attr_df.at['edge_load_centrality', area] = edge_load_data
        # save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # 9. CLUSTERING: geometric average of the subgraph edge weights
    if pd.isnull(attr_df.loc['clustering*', area]):
        clustering = nx.clustering(new_diG)
        attr_df = dict_data(clustering, shp_path, 'clustering*', area, attr_df)
        # clustering_data = dict_data(clustering, shp_path, 'clustering')
        # attr_df.at['clustering*', area] = clustering_data

        clustering_weighted = nx.clustering(new_diG, weight='time')
        attr_df = dict_data(clustering_weighted, shp_path, 'clustering_w*', area, attr_df)
        # clustering_weighted_data = dict_data(clustering_weighted, shp_path, 'clustering_w')
        # attr_df.at['clustering_w*', area] = clustering_weighted_data

        # save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # NETWORK SHAPE ATTRIBUTES
    # 10. ECCENTRICITY: maximum distance from v to all other nodes in G
    # Radius/Diameter of graph: radius is minimum eccentricity/The diameter is the maximum eccentricity.
    if pd.isnull(attr_df.loc['eccentricity', area]):
        eccentricity = nx.algorithms.distance_measures.eccentricity(new_G)
        attr_df = dict_data(eccentricity, shp_path, 'eccentricity', area, attr_df)
        attr_df = dict_data(min(eccentricity.values()), shp_path, 'radius', area, attr_df)
        attr_df = dict_data(max(eccentricity.values()), shp_path, 'diameter', area, attr_df)
        # eccentricity_data = dict_data(eccentricity, shp_path, 'eccentricity')
        # attr_df.at['eccentricity', area] = eccentricity_data
        # attr_df.at['radius', area] = nx.radius(new_G)
        # attr_df.at['diameter', area] = nx.diameter(new_G)
        # save_attr_df(attr_df, study_area_dir)

    # Center: center is the set of nodes with eccentricity equal to radius.
    # if pd.isnull(attr_df.loc['center_nodes', area]):
    #     center = nx.algorithms.distance_measures.center(new_G)
    #     attr_df.at['center_nodes', area] = center
    #     print(datetime.datetime.now(), 'Center nodes: ' + str(center))
    #     save_attr_df(attr_df, study_area_dir)
    # # Periphery: set of nodes with eccentricity equal to the diameter.
    # if pd.isnull(attr_df.loc['periphery_nodes', area]):
    #     periphery = nx.algorithms.distance_measures.periphery(new_G)
    #     attr_df.at['periphery_nodes', area] = periphery
    #     print(datetime.datetime.now(), 'Periphery nodes: ' + str(periphery))
    #     save_attr_df(attr_df, study_area_dir)
    # # Baryecnter: subgraph that minimizes the function sum(w(u,v))
    # if pd.isnull(attr_df.loc['barycenter_nodes', area]):
    #     barycenter = nx.algorithms.distance_measures.barycenter(new_G, weight='time')
    #     attr_df.at['barycenter_nodes', area] = barycenter
    #     print(datetime.datetime.now(), 'Baryenter nodes: ' + str(barycenter))
    #     save_attr_df(attr_df, study_area_dir)

    # 'network_distance'
    if pd.isnull(attr_df.loc['network_distance', area]):
        nodes_list = new_G.edges.data('length', default=1)
        total_len = 0
        for nod in nodes_list:
            total_len += nod[2]

        attr_df = dict_data(total_len, shp_path, 'network_distance', area, attr_df)
        # attr_df.at['network_distance', area] = total_len
        # print(datetime.datetime.now(), 'Total network distance: ' + str(total_len))
        # save_attr_df(attr_df, study_area_dir)

    # ----------------------------------------------------------------------
    # 11. OSMnx stats module
    # if not os.path.isfile(str(shp_path) + '/' + 'stats_basic.pkl'):
    #     print('Calculating basic_stats')
    #     new_G.graph['crs'] = 'epsg:2056'
    #     new_G.graph['name'] = str(area) + '_MultiDiGraph'
    #     basic_stats = ox.basic_stats(new_G, area=study_area_shp.area, clean_intersects=True, tolerance=15,
    #                                  circuity_dist='euclidean')
    #     with open(str(shp_path) + '/stats_basic.pkl', 'wb') as f:
    #         pickle.dump(basic_stats, f, pickle.HIGHEST_PROTOCOL)
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
        if pd.isnull(attr_df.loc[basic_stats_list[attr], area]):
            file = open(str(shp_path) + "/stats_basic.pkl", 'rb')
            stats_basic_dict = pickle.load(file)
            attr_df = dict_data(stats_basic_dict[attr], shp_path, basic_stats_list[attr], area, attr_df)
            # attr_df.at[basic_stats_list[attr], area] = stats_basic_dict[attr]
            # print(datetime.datetime.now(), 'Attribute ' + str(basic_stats_list[attr]) + ': '
            #       + str(stats_basic_dict[attr]))
            # save_attr_df(attr_df, study_area_dir)

    print(datetime.datetime.now(), 'Attributes stored in attr_df successfully but not saved yet.')
    print('----------------------------------------------------------------------')

    return attr_df, attributes


def take_avg(attr_df, study_area_dir, attributes=None):
    # take avg of cells (if possible)
    attr_avg_df = attr_df.copy(deep=True)

    for i in range(len(attr_df)):
        # print(attr_df.index[i], i)
        for column in attr_df.columns:
            try:
                # value = attr_df.iloc[i][column]
                # INTEGER ATTRIBUTES
                if attr_df.index[i] in ['n_nodes', 'n_edges', 'population', 'trips',
                                        'radius', 'diameter', 'n_intersection', 'n_street']:
                # if isinstance(value, int):
                    str_val = attr_df.iloc[i][column]
                    int_val = int(float(str_val))
                    attr_df.at[attr_df.index[i], column] = int_val
                    attr_avg_df.at[attr_avg_df.index[i], column] = int_val

                # FLOAT ATTRIBUTES
                elif attr_df.index[i] in ['network_distance', 'area', 'avg_degree', 'avg_edge_density',
                                          'avg_shortest_path_duration', 'streets_per_node', 'node_d_km',
                                          'intersection_d_km', 'edge_d_km', 'street_d_km', 'circuity_avg']:
                # elif isinstance(value,float):
                    str_val = attr_df.iloc[i][column]
                    flo_val = float(str_val)
                    attr_df.at[attr_df.index[i], column] = flo_val
                    attr_avg_df.at[attr_avg_df.index[i], column] = flo_val

                # LIST ATTRIBUTES: [n, min, max, avg]
                elif attr_df.index[i] in ['degree_centrality', 'avg_degree_connectivity', 'node_betweenness*',
                                          'edge_betweenness', 'lim_edge_betweenness', 'node_load_centrality',
                                          'edge_load_centrality',
                                          'clustering*', 'eccentricity', 'node_closeness_time*',
                                            'node_closeness_length*', 'avg_neighbor_degree',
                                          'node_straightness', 'clustering_w*', 'btw_home_trip_production',
                                          'btw_empl_trip_generation', 'btw_acc_trip_generation', 'btw_acc_trip_production']:
                # elif type(value) is list:
                    str_val = attr_df.iloc[i][column]
                    list_val = ast.literal_eval(str_val)
                    attr_df.at[attr_df.index[i], column] = list_val
                    attr_avg_df.at[attr_avg_df.index[i], column] = list_val[3]

                # elif attr_df.index[i] in ['center_nodes', 'periphery_nodes', 'barycenter_nodes']:
                    # str_val = attr_df.iloc[i][column]
                    # flo_val = int(str_val[2:-2])
                    # attr_avg_df.drop(attr_avg_df.index[i])
                    # attr_df_avg.at[attr_df_avg.index[i], column] = flo_val
            except:
                attr_df.at[attr_df.index[i], column] = None
                attr_avg_df.at[attr_avg_df.index[i], column] = None

    # reindex dataframe
    # new_index = [ 'n_nodes',             # stats_basic
    #               'n_edges',            # stats_basic
    #               'network_distance',   # stats_basic
    #               'area',
    #               'population',
    #               'trips',
    #               'n_intersection',     # stats_basic
    #               'n_street',           # stats_basic
    #               'streets_per_node',   # stats_basic
    #               'node_d_km',          # stats_basic
    #               'intersection_d_km',  # stats_basic
    #               'edge_d_km',          # stats_basic
    #               'street_d_km',        # stats_basic
    #               'circuity_avg',       # stats_basic
    #               'avg_degree',
    #               'avg_neighbor_degree',
    #               'degree_centrality',
    #               'avg_degree_connectivity',
    #               'avg_edge_density',
    #               'avg_shortest_path_duration',
    #               'node_betweenness*',
    #               'edge_betweenness',
    #               'lim_edge_betweenness',
    #               'btw_home_trip_production',
    #               'btw_empl_trip_generation',
    #               'btw_acc_trip_generation',
    #               'btw_acc_trip_production',
    #               'node_straightness',
    #               'node_closeness_time*',
    #               'node_closeness_length*',
    #               'node_load_centrality',
    #               'edge_load_centrality',
    #               'clustering*',
    #               'clustering_w*',
    #               'eccentricity',
    #               'radius',
    #               'diameter',
    #               'center_nodes',
    #               'periphery_nodes',
    #               'barycenter_nodes']
    # attr_df = attr_df.reindex(new_index)
    # attr_avg_df = attr_avg_df.reindex(new_index)

    if attributes:
        attr_df = attr_df.reindex(attributes)
        attr_avg_df = attr_avg_df.reindex(attributes)

    # sort both dataframes
    attr_df = attr_df.sort_values(by='network_distance', ascending=False, axis=1)
    attr_avg_df = attr_avg_df.sort_values(by='network_distance', ascending=False, axis=1)

    # delete some columns and rows in avg table
    attr_avg_df = attr_avg_df.drop(index=['center_nodes', 'periphery_nodes', 'barycenter_nodes', 'clustering_w*'],
                                   columns=['bern_large', 'zurich_large', 'lausanne_lake'])

    # create a transpose of the matrix
    attr_avg_dfT = attr_avg_df.transpose()

    # save the three dataframes to csv
    save_attr_df(attr_df, study_area_dir, attr_avg_df, attr_avg_dfT)

    # return attr_df, attr_avg_df


# study_area_dir = r'C:\Users\Ion\TFM\data\study_areas'
# topology_attributes(study_area_dir)
    # ideas for attributes:
    # concentration of nodes/edges in different grids
    # avg trip distance
    # distance of cg from home cluster to work/education cluster
    # take formulas from documentation (reference)
