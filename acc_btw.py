import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
from scipy import spatial
import pickle
from shapely.geometry import Point, Polygon, LinearRing
import shapely.geometry as geo
import datetime
import os
import random
from matplotlib import pyplot as plt
import math
import ntpath
import ast
import igraph as ig
import copy
from scipy.optimize import curve_fit

# from graph_analysis import dict_data, create_igraph #not necessary to import as this function is called from graph_analysis
from network_graph import check_iso_graph
from population_distribution import create_distrib


def cutting_graph(study_area_dir, graph_file, area):
    print(datetime.datetime.now(), 'Creating graph for area ...')

    file = open(str(graph_file) + "/ch_nodes_dict2056.pkl", 'rb')
    nodes_dict = pickle.load(file)

    shp_path = str(study_area_dir) + "/" + str(area)
    study_area_shp = gpd.read_file(str(shp_path) + "/" + str(area) + ".shp").iloc[0]['geometry']


    G = nx.read_gpickle(str(graph_file) + '/ch_MultiDiGraph_bytime_largest.gpickle')
    diG = nx.read_gpickle(str(graph_file) + '/ch_DiGraph_bytime_largest.gpickle')

    new_G = cut_graph(G, shp_path, 'MultiDiGraph', area, study_area_shp)
    new_diG = cut_graph(diG, shp_path, 'DiGraph', area, study_area_shp)

    poly_line = LinearRing(study_area_shp.exterior)
    poly_line_offset = poly_line.buffer(5000, resolution=16, join_style=2, mitre_limit=1)
    study_area_shp5k = Polygon(list(poly_line_offset.exterior.coords))

    new_G5k = cut_graph(G, shp_path, 'MultiDiGraph5k', area, study_area_shp5k)

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
    export_shp(new_G5k, str(shp_path) + "/" + str(area) + "_network_5k.shp")
    print('----------------------------------------------------------------------')


def dict_data(dicti, study_area_dir, attrib_name, area, area_series, filename=None):
    shp_path = str(study_area_dir) + "/" + str(area)
    attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes',
                          dtype=object)
    # if '*' in attrib_name: #this means DiGraph was used to compute attribute instead of MultiDiGraph
    if not filename:
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

    # area_series[area] = value
    attr_df.at[attrib_name, area] = value
    save_attr_df(attr_df, study_area_dir, area)
    print(datetime.datetime.now(), 'Attribute ' + str(attrib_name) + ' calculated: ' + str(value))
    return area_series


def save_attr_df(attr_df, study_area_dir, area, attr_avg_df=None, attr_avg_dfT=None):
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


def straightness(nodes_dict, new_G, nodes_list=None):
    print(datetime.datetime.now(), 'Calculating node_straightness of graph ...')
    if nodes_list:
        nodes_list = nodes_list
    else:
        nodes_list = []
        for node in list(new_G.nodes):
            nodes_list.append(str(node))
        # nodes_list = list(new_G.nodes)
    print(datetime.datetime.now(), 'Number of nodes to route: ' + str(len(nodes_list)))
    g = create_igraph(new_G)
    n = 0
    node_straightness = {}
    # for i in g.vs['name']:
    for i in nodes_list:
        # print(datetime.datetime.now(), n)
        # if n % 1 == 0:
        #     print(datetime.datetime.now(), n)
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


def take_avg(study_area_dir, area, attributes=None):
    attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes',
                          dtype=object)
    # take avg of cells (if possible)
    attr_avg_df = attr_df.copy(deep=True)

    for i in range(len(attr_df)):
        # print(attr_df.index[i], i)
        for column in attr_df.columns:
            try:
                # value = attr_df.iloc[i][column]
                # INTEGER ATTRIBUTES
                if attr_df.index[i] in ['n_nodes', 'n_edges', 'population', 'trips',
                                        'radius', 'diameter', 'n_intersection', 'n_street', 'CarPt_users']:
                # if isinstance(value, int):
                    str_val = attr_df.iloc[i][column]
                    int_val = int(float(str_val))
                    attr_df.at[attr_df.index[i], column] = int_val
                    attr_avg_df.at[attr_avg_df.index[i], column] = int_val

                # FLOAT ATTRIBUTES
                elif attr_df.index[i] in ['network_distance', 'area', 'avg_degree', 'avg_edge_density',
                                          'avg_shortest_path_duration', 'streets_per_node', 'node_d_km',
                                          'intersection_d_km', 'edge_d_km', 'street_d_km', 'circuity_avg',
                                          'avg_link_time', 'efficiency', 'population_density', 'population_gini']:
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

                # STRING ATTRIBUTES:
                elif attr_df.index[i] in ['area_type']:
                    attr_avg_df.at[attr_avg_df.index[i], column] = attr_df.iloc[i][column]


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

    if attributes is not None:
        attr_df = attr_df.reindex(attributes)
        attr_avg_df = attr_avg_df.reindex(attributes)

    # sort both dataframes
    attr_df = attr_df.sort_values(by='network_distance', ascending=False, axis=1)
    attr_avg_df = attr_avg_df.sort_values(by='network_distance', ascending=False, axis=1)

    # delete some columns and rows in avg table
    attr_avg_df = attr_avg_df.drop(index=['center_nodes', 'periphery_nodes', 'barycenter_nodes', 'clustering_w*'],
                                   columns=['bern_large', 'zurich_large', 'lausanne_lake', 'test_area'])

    # create a transpose of the matrix
    attr_avg_dfT = attr_avg_df.transpose()

    # save the three dataframes to csv
    save_attr_df(attr_df, study_area_dir, area, attr_avg_df, attr_avg_dfT)

    # return attr_df, attr_avg_df


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def dist_decay_funct(area_path, n_bins):
    # area = 'zurich_kreis'
    # area_path = r'C:/Users/Ion/TFM/data/study_areas/' + str(area)
    pop_path = str(area_path) + '/population_db'

    # In case there are more than 1 population_plans file to concatenate:
    # data = []
    # file_list = onlyfiles = [f for f in listdir(pop_path) if isfile(join(pop_path, f))]
    # for filename in file_list:
    #     if 'population_plans_' in filename and '.csv' in filename:
    #         plans_file = pd.read_csv(str(pop_path) + '/' + str(filename))
    #         plans_data = list(plans_file['trav_time'])
    #         if len(data) > 0:
    #             data = data + plans_data

    plans = pd.read_csv(str(pop_path) + '/population_plans_0.csv', low_memory=False)
    data = list(plans['trav_time'])

    time_data = []
    for t in data:
        try:
            time = get_sec(t)
            if time > 0:
                time_data.append(time)
        except:
            continue

    # n_bins = 1000
    bins = np.linspace(math.ceil(min(time_data)),
                       math.floor(max(time_data)),
                       n_bins)  # fixed number of bins

    hist = np.histogram(time_data, bins=bins)
    y = hist[0]/sum(hist[0])
    x = []
    zeros = 0
    interval = hist[1][2] - hist[1][1]
    for i in range(1, len(y)):
        x_lab = i*interval - interval/2
        x.append(x_lab)
        # if x_lab > 1500:
        #     y = y[:i]
        #     x = np.array(x[:i])
        #     break
        if y[i] < 0.001:
            zeros += 1
            if zeros == 20:
                y = y[:i]
                x = np.array(x[:i])
                break
    # print(len(x), len(y))
    # plt.plot(x, y, 'o', color='black')

    # ------------------------------------------------------
    # define type of function to search
    def model_func(x, a, k, b):
        return a * np.exp(-k * x) + b

    # curve fit
    p0 = (1., 1.e-5, 1.)  # starting search koefs
    opt, pcov = curve_fit(model_func, x, y, p0=(1., 1.e-5, 1.), maxfev=5000)

    # this values have to be sent back to btw function to build calibrated decay function:
    a, k, b = opt

    # test result
    # x2 = np.linspace(1, max(x), len(x))
    # y2 = model_func(x2, a, k, b)
    # fig, ax = plt.subplots()
    # ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.9f x} %+.3f$' % (a, k, b))
    # ax.plot(x, y, 'bo', label='data with noise')
    # ax.legend(loc='best')
    # plt.title('Distribution of trips in study area')
    # plt.xlabel('trip length (in seconds)')
    # plt.ylabel('count (normalized)')
    # plt.show()

    return a, k, b


def impedance_function(area_path, path_time=None, a=None, c=None, b=None):
    if a is None:
        # calibrates the distance decay function taking the correspondant travel times of the study area
        a, c, b = dist_decay_funct(area_path, n_bins=1000)
        # a,b,c=[0,0,0]
        return a, c, b

    #calibrated function based in trip travel times
    f_tt = (a * math.exp(c * path_time) + b)
    # f_tt = math.exp(-0.05 * path_time)
    return f_tt


def btw_acc(new_G, chG, study_area_dir, area, nodes_dict, area_series, grid_size):
    # import nodes into study area: new_G.nodes()
    # import graph of full ch and transform into igraph
    # iterate over all pair of nodes in the study areas nodes by a maximum time
    # file = open(r'C:/Users/Ion/TFM/data/network_graphs/ch_nodes_dict2056.pkl', 'rb')
    # nodes_dict = pickle.load(file)
    # area = 'zurich_kreis'
    # chG = nx.read_gpickle(r'C:/Users/Ion/TFM/data/study_areas/' + str(area) + '/' + str(area) + '_MultiDiGraph5k_largest.gpickle')
    # new_G = nx.read_gpickle(r'C:/Users/Ion/TFM/data/study_areas/' + str(area) + '/' + str(area) + '_MultiDiGraph_largest.gpickle')
    # area_path = r'C:/Users/Ion/TFM/data/study_areas/' + str(area)


    print(datetime.datetime.now(), 'Calculating lim_edge_betweenness of graph ...')

    area_path = str(study_area_dir) + "/" + str(area)
    time_lim = 1200  # seconds
    # grid_size = 2000  # meters
    g = create_igraph(chG)

    # calibrates the distance decay function taking the correspondant travel times of the study area
    a, c, b = impedance_function(area_path, path_time=None, a=None, c=None, b=None)

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

    print(datetime.datetime.now(), 'Finding respective nodes of each grid ...')
    tree = spatial.KDTree(G_lonlat)
    m_df['nodes_in_grid'] = np.nan
    m_df['nodes_in_grid'] = m_df['nodes_in_grid'].astype('object')
    for index, row in m_df.iterrows():
        # print(index)
        # This fixes every grid starting node with the centroids closest node id:
        centroid = Point(row['centroid'][0], row['centroid'][1])

        nn = tree.query(centroid)
        coord = G_lonlat[nn[1]]
        closest_node_id = int(list(nodes_dict.keys())[list(nodes_dict.values()).index((coord[0], coord[1]))])
        m_df.at[index, 'closest_node'] = int(closest_node_id)

        # This concatenates all the nodes on each grid, to later assign randomly as a starting point
        grid = row['geometry']

        nodes_in_grid = []
        for node in G_nodes:
            network_node = Point(new_G.nodes[node]['x'], new_G.nodes[node]['y'])
            in_grid = grid.contains(network_node)
            if in_grid:
                nodes_in_grid.append(node)
        # m_df.at[index, 'nodes_in_grid'] = list([nodes_in_grid])
        m_df.at[index, 'nodes_in_grid'] = nodes_in_grid
    print(datetime.datetime.now(), 'Closest node of every grid found.')

    if not os.path.isfile(str(area_path) + "/" + str(area) + '_' + "facility_distribution_" +
                  str(grid_size) + "gs.csv"):
        # add columns of ACC empl and ACC pop respectively to the df and export as shapefile:
        # for facil in ['pop', 'empl', 'opt']:
        for facil in ['pop', 'empl']:
            # for facil in ['pop']:
            print(datetime.datetime.now(), 'Calculating ' + str(facil) + ' accessibility for each grid area ...')
            for index, row in m_df.iterrows():
                # print(index)
                if len(row['nodes_in_grid']) > 0:
                    j = str(random.choice(row['nodes_in_grid']))
                else:
                    j = str(int(row['closest_node']))
                facil_j = row[facil]  # iterate over pop, empl, opt
                acc_list = []
                for index2, row2 in m_df.iterrows():
                    if len(row2['nodes_in_grid']) > 0:
                        k = str(random.choice(row2['nodes_in_grid']))
                    else:
                        k = str(int(row2['closest_node']))
                    if j == k:
                        continue
                    # find sp and sum time
                    paths = g.get_shortest_paths(v=j, to=k, weights='time', mode='OUT', output="epath")
                    path_time = 0
                    for i in paths[0]:
                        path_time += g.es[i]['time']

                    # define impedance function
                    # try: #calibrated function based in trip travel times
                    # acc = facil_j * (a * math.exp(c * path_time) + b)
                    # except:
                    f_tt = impedance_function(area_path, path_time, a, c, b)
                    acc = facil_j * f_tt
                    acc_list.append(acc)
                m_df.at[index, 'acc_' + str(facil)] = sum(acc_list)
            print(datetime.datetime.now(), 'Computation of ' + str(facil) + ' accessibility finished.')

        m_df.to_csv(str(area_path) + "/" + str(area) + '_' + "facility_distribution_" + str(grid_size) + "gs.csv", sep=",")
        m_gdf = gpd.GeoDataFrame(m_df)
        m_gdf = m_gdf.drop(['centroid', 'nodes_in_grid'], axis=1)
        m_gdf.to_file(str(area_path) + "/" + str(area) + '_' + "facility_distribution_" +
                      str(grid_size) + "gs.shp")
    else:
        m_df = pd.read_csv(str(area_path) + "/" + str(area) + '_' + "facility_distribution_" + str(grid_size) + "gs.csv", sep=",")
        print('Load csv found')

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
            if len(row_j['nodes_in_grid']) > 0:
                j = str(random.choice(row_j['nodes_in_grid']))
            else:
                j = str(int(row_j['closest_node']))

            acc_j = row_j['acc_' + str(other_facil)]  # iterate over pop, empl
            facil_j = row_j[facil]
            if acc_j == 0:
                continue

            for index2, row_k in m_df.iterrows():
                if len(row_k['nodes_in_grid']) > 0:
                    k = str(random.choice(row_k['nodes_in_grid']))
                else:
                    k = str(int(row_k['closest_node']))

                facil_k = row_k[other_facil]
                facil2_k = row_k[facil]
                if j == k:
                    continue

                # Compute shortest paths for j, k
                paths = g.get_shortest_paths(v=j, to=k, weights='time', mode='OUT', output="epath")
                path_time = 0
                way_sp = {}  # stores the number of occurrencies of each link in the shortest paths: d[way_id] = int
                for path in paths:
                    for i in path:
                        path_time += g.es[i]['time']
                        way_id = g.es[i]['name']
                        if not way_sp.get(way_id):
                            way_sp[way_id] = 1
                        else:
                            way_sp[way_id] += 1

                # Compute weight and centrality value for every link in paths
                # define impedance function
                # try: #calibrated function based in trip travel times
                # f_tt = (a * math.exp(c * path_time) + b)
                # print(f_tt)
                # except:
                # f_tt = math.exp(-0.05 * path_time)

                f_tt = impedance_function(area_path, path_time, a, c, b)

                weight_C_facil = facil_j * ((facil_k * f_tt) / acc_j)
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
        if os.path.isfile(str(area_path) + "/" + str(area) + "_btw_acc_" + str(grid_size) + ".shp"):
            links_gdf = gpd.read_file(str(area_path) + "/" + str(area) + "_btw_acc_" + str(grid_size) + ".shp")
        else:
            links_gdf = gpd.read_file(str(area_path) + "/" + str(area) + "_network_5k.shp")
            links_gdf['btw_check'] = 0

        links_gdf['btw_' + str(facil)] = 0
        links_gdf['btw_Acc_' + str(facil)] = 0
        for link in list(cPop_li):
            cPop_li[link] = sum(cPop_li[link]) / tot_pop
            cApop_li[link] = sum(cApop_li[link]) / tot_acc

            # update gdf of network with obtained values
            ix = links_gdf.index[links_gdf['way_id'] == link][0]

            links_gdf.loc[ix, 'btw_' + str(facil)] = cPop_li[link]
            links_gdf.loc[ix, 'btw_Acc_' + str(facil)] = cApop_li[link]
            links_gdf.loc[ix, 'btw_check'] = 1

        # links_gdf.to_file(str(area_path) + "/" + str(area) + "_btw_acc.shp")
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
    links_gdf.to_file(str(area_path) + "/" + str(area) + "_btw_acc_" + str(grid_size) + ".shp")

    # Export dictionaries and save values in attributes table
    dict_data(cPop_li, study_area_dir, 'btw_home_trip_production', area, area_series, 'btw_home_trip_production' + str(grid_size) + 'gs')
    dict_data(cEmpl_li, study_area_dir, 'btw_empl_trip_generation', area, area_series, 'btw_empl_trip_generation' + str(grid_size) + 'gs')

    dict_data(cApop_li, study_area_dir, 'btw_acc_trip_generation', area, area_series, 'btw_acc_trip_generation' + str(grid_size) + 'gs')
    dict_data(cAempl_li, study_area_dir, 'btw_acc_trip_production', area, area_series, 'btw_acc_trip_production' + str(grid_size) + 'gs')

    print(datetime.datetime.now(), 'Betweenness-accessibility metrics computation finished.')
# return edge_btw_acc


def eccentricity(G, v=None, sp=None):
    """Returns the eccentricity of nodes in G.

    The eccentricity of a node v is the maximum distance from v to
    all other nodes in G.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    v : node, optional
       Return value of specified node

    sp : dict of dicts, optional
       All pairs shortest path lengths as a dictionary of dictionaries

    Returns
    -------
    ecc : dictionary
       A dictionary of eccentricity values keyed by node.
    """
#    if v is None:                # none, use entire graph
#        nodes=G.nodes()
#    elif v in G:               # is v a single node
#        nodes=[v]
#    else:                      # assume v is a container of nodes
#        nodes=v
    order = G.order()

    e = {}
    for n in G.nbunch_iter(v):
        if sp is None:
            length = nx.single_source_shortest_path_length(G, n)
            L = len(length)
        else:
            try:
                length = sp[n]
                L = len(length)
            except TypeError:
                raise nx.NetworkXError('Format of "sp" is invalid.')
        if L != order:
            if G.is_directed():
                msg = ('Found infinite path length because the digraph is not'
                       ' strongly connected')
            else:
                msg = ('Found infinite path length because the graph is not'
                       ' connected')
            raise nx.NetworkXError(msg)

        e[n] = max(length.values())

    if v in G:
        return e[v]  # return single value
    else:
        return e


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


# btw_acc(new_G, chG, study_area_dir, area, nodes_dict, area_series, grid_size=500)
# def opt_btw(new_G, chG, study_area_dir, area, nodes_dict, area_series):
#     g = create_igraph(chG)
#
#     for j in list(new_G.nodes):
#         for k in list(new_G.nodes):
#             if j == k:
#                 continue
#
#             paths = g.get_shortest_paths(v=j, to=k, weights='time', mode='OUT', output="epath")
#             path_time = 0
#             way_sp = {}  # stores the number of occurrencies of each link in the shortest paths: d[way_id] = int
#             for path in paths:
#                 for i in path:
#                     path_time += g.es[i]['time']
#                     way_id = g.es[i]['name']
#                     start_node = i.source
#                     end_node = i.target
#                     if not way_sp.get(way_id):
#                         way_sp[way_id] = 1
#                     else:
#                         way_sp[way_id] += 1
#
#
#     for index, row_j in m_df.iterrows():
#         if len(row_j['nodes_in_grid']) > 0:
#             j = str(random.choice(row_j['nodes_in_grid']))
#         else:
#             j = str(int(row_j['closest_node']))
#
#         acc_j = row_j['acc_' + str(other_facil)]  # iterate over pop, empl
#         facil_j = row_j[facil]
#         if acc_j == 0:
#             continue
#
#         for index2, row_k in m_df.iterrows():
#             if len(row_k['nodes_in_grid']) > 0:
#                 k = str(random.choice(row_k['nodes_in_grid']))
#             else:
#                 k = str(int(row_k['closest_node']))
#
#             facil_k = row_k[other_facil]
#             facil2_k = row_k[facil]
#             if j == k:
#                 continue
#
#             # Compute shortest paths for j, k
#             paths = g.get_shortest_paths(v=j, to=k, weights='time', mode='OUT', output="epath")
#             path_time = 0
#             way_sp = {}  # stores the number of occurrencies of each link in the shortest paths: d[way_id] = int
#             for path in paths:
#                 for i in path:
#                     path_time += g.es[i]['time']
#                     way_id = g.es[i]['name']
#                     if not way_sp.get(way_id):
#                         way_sp[way_id] = 1
#                     else:
#                         way_sp[way_id] += 1
#
#             weight_C_facil = facil_j * ((facil_k * f_tt) / acc_j)
#             weight_Ca_facil = facil2_k * f_tt
#
#             for link in list(way_sp):
#                 c_pop = (way_sp[link] / len(paths)) * weight_C_facil
#                 cA_pop = (way_sp[link] / len(paths)) * weight_Ca_facil
#                 if not cPop_li.get(link):
#                     cPop_li[link] = [c_pop]
#                     cApop_li[link] = [cA_pop]
#                 else:
#                     a_list = cPop_li[link]
#                     a_list.append(c_pop)
#                     cPop_li[link] = a_list
#
#                     b_list = cApop_li[link]
#                     b_list.append(cA_pop)
#                     cApop_li[link] = b_list
#
#     # Finally, for every link, compute sum of every pair of nodes, obtaining betweenness values
#     # Define absolute values for normalization:
#     tot_pop = m_df[facil].sum()
#     tot_acc = m_df['acc_' + str(facil)].sum()
#
#     # Load geodataframe with every link of network to add btw-acc values
#     if os.path.isfile(str(area_path) + "/" + str(area) + "_btw_acc.shp"):
#         links_gdf = gpd.read_file(str(area_path) + "/" + str(area) + "_btw_acc.shp")
#     else:
#         links_gdf = gpd.read_file(str(area_path) + "/" + str(area) + "_network_5k.shp")
#         links_gdf['btw_check'] = 0
#
#     links_gdf['btw_' + str(facil)] = 0
#     links_gdf['btw_Acc_' + str(facil)] = 0
#     for link in list(cPop_li):
#         cPop_li[link] = sum(cPop_li[link]) / tot_pop
#         cApop_li[link] = sum(cApop_li[link]) / tot_acc
#
#         # update gdf of network with obtained values
#         ix = links_gdf.index[links_gdf['way_id'] == link][0]
#
#         links_gdf.loc[ix, 'btw_' + str(facil)] = cPop_li[link]
#         links_gdf.loc[ix, 'btw_Acc_' + str(facil)] = cApop_li[link]
#         links_gdf.loc[ix, 'btw_check'] = 1
