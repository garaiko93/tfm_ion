import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
from scipy import spatial
import pickle
from shapely.geometry import Point, Polygon, LinearRing
import datetime
import os
import random
from matplotlib import pyplot as plt
import math
from scipy.optimize import curve_fit

from graph_analysis import dict_data, create_igraph
from population_distribution import create_distrib


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

    plans = pd.read_csv(str(pop_path) + '/population_plans_0.csv')
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
            if zeros == 10:
                y = y[:i]
                x = np.array(x[:i])
                break
    print(len(x), len(y))
    plt.plot(x, y, 'o', color='black')

    # ------------------------------------------------------
    # define type of function to search
    def model_func(x, a, k, b):
        return a * np.exp(-k * x) + b

    # curve fit
    p0 = (1., 1.e-5, 1.)  # starting search koefs
    opt, pcov = curve_fit(model_func, x, y, p0)

    # this values have to be sent back to btw function to build calibrated decay function:
    a, k, b = opt
    return a, k, b

    # test result
    # x2 = np.linspace(1, max(x), 250)
    # y2 = model_func(x2, a, k, b)
    # fig, ax = plt.subplots()
    # ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a, k, b))
    # ax.plot(x, y, 'bo', label='data with noise')
    # ax.legend(loc='best')
    # plt.title('Distribution of trips in study area')
    # plt.xlabel('trip length (in seconds)')
    # plt.ylabel('count (normalized)')
    # plt.show()

def btw_acc(new_G, chG, area_path, area, nodes_dict, attr_df, grid_size=500):
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
    time_lim = 1200  # seconds
    grid_size = 300  # meters
    g = create_igraph(chG)

    # calibrates the distance decay function taking the correspondant travel times of the study area
    a, k, b = dist_decay_funct(area_path, n_bins=1000)

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
        m_df.at[index, 'nodes_in_grid'] = list([nodes_in_grid])
    print(datetime.datetime.now(), 'Closest node of every grid found.')

    # add columns of ACC empl and ACC pop respectively to the df and export as shapefile:
    # for facil in ['pop', 'empl', 'opt']:
    for facil in ['pop', 'empl']:
        # for facil in ['pop']:
        print(datetime.datetime.now(), 'Calculating ' + str(facil) + ' accessibility for each grid area ...')
        for index, row in m_df.iterrows():
            # j = str(int(row['closest_node']))
            j = random.choice(row['nodes_in_grid'][0])
            facil_j = row[facil]  # iterate over pop, empl, opt
            acc_list = []
            for index2, row2 in m_df.iterrows():
                # k = str(int(row2['closest_node']))
                k = random.choice(row2['nodes_in_grid'][0])
                if j == k:
                    continue
                # find sp and sum time
                paths = g.get_shortest_paths(v=j, to=k, weights='time', mode='OUT', output="epath")
                path_time = 0
                for i in paths[0]:
                    path_time += g.es[i]['time']

                # define impedance function
                try: #calibrated function based in trip travel times
                    acc = facil_j * (a * math.exp(k * path_time) + b)
                except:
                    acc = facil_j * math.exp(-0.05 * path_time)
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
            # j = str(int(row_j['closest_node']))
            j = random.choice(row_j['nodes_in_grid'][0])
            acc_j = row_j['acc_' + str(other_facil)]  # iterate over pop, empl
            facil_j = row_j[facil]
            if acc_j == 0:
                continue

            for index2, row_k in m_df.iterrows():
                # k = str(int(row_k['closest_node']))
                k = random.choice(row_k['nodes_in_grid'][0])
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
                try: #calibrated function based in trip travel times
                    f_tt = (a * math.exp(k * path_time) + b)
                except:
                    f_tt = math.exp(-0.05 * path_time)
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
        if os.path.isfile(str(area_path) + "/" + str(area) + "_btw_acc.shp"):
            links_gdf = gpd.read_file(str(area_path) + "/" + str(area) + "_btw_acc.shp")
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