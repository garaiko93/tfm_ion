import gzip
import re
import pandas as pd
import pickle
import geopandas as gpd
import shapely.geometry as geo
import networkx as nx
import copy
import os
import datetime
from shutil import copyfile
from progressbar import Percentage, ProgressBar, Bar, ETA


# Identify the largest component and the "isolated" nodes
def check_iso_graph(G, out_path, filename):
    print(datetime.datetime.now(), 'Original ' + str(filename) + ' has: ' + str(
        len([len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)])) + ' island with '
          + str(G.number_of_nodes()) + '/' + str(G.number_of_edges()) + ' (Nnodes/Nedges)')

    components = list(nx.strongly_connected_components(G))  # list because it returns a generator
    components.sort(key=len, reverse=True)
    longest_networks = []
    if len(components) > 1:
        for i in range(0, min(len(components), 10)):
            net = components[i]
            longest_networks.append(len(net))
        largest = components.pop(0)
        isolated = set(g for cc in components for g in cc)
        num_isolated = G.order() - len(largest)

        # keep only the largest island of the original graph (so all nodes are reachable in the graph)
        # remove isolated nodes from G
        G.remove_nodes_from(isolated)
        # export final graph only containing largest island from the network to file
        nx.write_gpickle(G, str(out_path) + "/" + str(filename) + "_largest.gpickle")

        print(datetime.datetime.now(), 'N isolated nodes: ' + str(num_isolated))
        print(datetime.datetime.now(), '10 largest networks (nodes): ' + str(longest_networks))

        print(datetime.datetime.now(), 'Resulting ' + str(filename) + ' has: ' + str(
            len([len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)])) + ' island with '
              + str(G.number_of_nodes()) + '/' + str(G.number_of_edges()) + ' (Nnodes/Nedges)')
    else:
        largest = components[0]
        isolated = []
        nx.write_gpickle(G, str(out_path) + "/" + str(filename) + ".gpickle")
        copyfile(str(out_path) + "/" + str(filename) + ".gpickle",
                 str(out_path) + "/" + str(filename) + "_largest.gpickle")

    return G, isolated, largest

def create_graph(G,edges_list,out_path,filename,nodes_dict):
    # introduce every way as edge with attributes of time and new_id
    for start, end, time, way_id, modes, length in edges_list:
        G.add_edge(start, end, time=time, way_id=way_id, modes=modes, length=length)

    # add attributes of coordinates to each node
    for node in list(G.nodes):
        G.nodes[node]['x'] = nodes_dict[str(node)][0]
        G.nodes[node]['y'] = nodes_dict[str(node)][1]
    # export graph of original network to file (without excluding any edge or island)
    nx.write_gpickle(G, str(out_path) + "/" + str(filename) + ".gpickle")
    G_isolated = copy.deepcopy(G)

    [G, isolated, largest] = check_iso_graph(G,out_path,filename)

    print('Input edges: ' + str(len(edges_list)))
    print('------------------------------------------------------------------------')
    return G, G_isolated, isolated, largest

# create shapefile with all nodes/edges excluded from the final graph (only for visual purpose)
def sw_nodes(way_id):
    coord_list = []
    start_node_id = ways_dict[str(way_id)][0]
    end_node_id = ways_dict[str(way_id)][1]
    for node in [start_node_id, end_node_id]:
        coord_list.append(nodes_dict[node])
    line = geo.LineString(coord_list)
    return line

def create_shp(edges, file_name):
    train = ch_ways_df[["start_node_id", "end_node_id", "time", "way_id", "modes"]]
    edges_df = pd.DataFrame.from_records(list(edges), columns=["start_node_id", "end_node_id"])
    intersected_df = pd.merge(edges_df, train, how='inner')

    intersected_df['geometry'] = intersected_df.apply(lambda row: sw_nodes(row['way_id']), axis=1)
    intersected_gdf = gpd.GeoDataFrame(intersected_df)
    intersected_gdf.to_file(str(out_path) + "/" + str(file_name) + ".shp")
    print('Shapefile exported as ' + str(file_name))

def parse_network(raw_file, out_path):
    if not os.path.exists(str(out_path)):
        os.makedirs(str(out_path))
        print('Directory created')
    else:
        print('Directory exists')
    print('------------------------------------------------------------------------')

    # #SPLIT OSM FILE IN FILES FOR: NODES, WAYS
    node_check = 0
    way_check = 0
    nodes_n = 0
    ways_n = 0
    lines_nodes = 0
    lines_links = 0
    lines_europe = 0

    # by reading the selected file line by line, for each xml element type this code splits the 3 of them in 3 different files for: NODES, WAYS and RELATIONS elements
    if os.path.isfile(str(out_path)+"\switzerland_network_nodes.xml.gz") == False and \
            os.path.isfile(str(out_path)+"\switzerland_network_ways.xml.gz") == False:
        print('Splitting Raw Network file into nodes-links ...')
        with gzip.open(raw_file) as f:
            with gzip.open(str(out_path)+"\switzerland_network_nodes.xml.gz", 'wb') as f1:
                with gzip.open(str(out_path)+"\switzerland_network_ways.xml.gz", 'wb') as f2:
                    for line in f:
                        lines_europe += 1
                        if b"<node" in line:
                            node_check = 1
                            nodes_n += 1
                        if node_check == 1:
                            f1.write(line)
                            lines_nodes += 1
                            if b"</node>" in line:
                                node_check = 0

                        if b"<link" in line:
                            way_check = 1
                            ways_n += 1
                        if way_check == 1:
                            f2.write(line)
                            lines_links += 1
                            if b"</link>" in line:
                                way_check = 0
        # export lines numbers
        lines = {}
        lines['lines_nodes'] = lines_nodes
        lines['lines_links'] = lines_links
        lines['lines_europe'] = lines_europe
        with open(str(out_path) + '/lines.pkl', 'wb') as f:
            pickle.dump(lines, f, pickle.HIGHEST_PROTOCOL)
    else:
        file = open(str(out_path) + "/lines.pkl", 'rb')
        lines = pickle.load(file)
        lines_nodes = lines['lines_nodes']
        lines_links = lines['lines_links']

    # SCREEN PRINT
    print('Lines of nodes: ' + str(lines_nodes))
    print('Lines of links: ' + str(lines_links))
    print('Lines in file: '+ str(lines_europe))
    print('Raw file splitted correctly in out_path')
    print('------------------------------------------------------------------------')
    # -----------------------------------------------------------------------------
    # NODES
    # -----------------------------------------------------------------------------
    if os.path.isfile(str(out_path) + "/ch_nodes_dict2056.pkl") == False:
        print('Parsing nodes from OSM xml file ...')
        nodes_dict = {}
        pbar = ProgressBar(widgets=[Bar('>', '[', ']'), ' ',
                                    Percentage(), ' ',
                                    ETA()], maxval=lines_nodes)
        with gzip.open(str(out_path) + "\switzerland_network_nodes.xml.gz") as f:
            #     reading line by line the 'nodes' file created at the beginning, data for each node fulfilling the conditions are stored for the output
            for line in pbar(f):
                if b"<node" in line:
                    # records the attributes of the element: node_id, latitude and longitude
                    m = re.search(rb'id="(.+)" x="([+-]?\d+(?:\.\d+)?)" y="([+-]?\d+(?:\.\d+)?)"', line)
                    if m is not None:
                        #         this is done to take only nodes that are contained in the ways filtered in the previous step
                        #         and taking into account that nodes ids are sorted in the OSM file
                        id = m.group(1).decode('utf-8')
                        lonlat = float(m.group(2)), float(m.group(3))
                        nodes_dict[id] = lonlat

        # EXPORT nodes_ch (dictionary) TO FILE
        with open(str(out_path) + '/ch_nodes_dict2056' + '.pkl', 'wb') as f:
            pickle.dump(nodes_dict, f, pickle.HIGHEST_PROTOCOL)
    else:
        file = open(str(out_path) + "/ch_nodes_dict2056.pkl", 'rb')
        nodes_dict = pickle.load(file)
        print('Nodes dictionary already exists, loaded')

    print('N_nodes in nodes_dict: ' + str(len(nodes_dict)))
    # -----------------------------------------------------------------------------
    # WAYS
    # -----------------------------------------------------------------------------
    if os.path.isfile(str(out_path) + "/ch_ways_dict.pkl") == False and \
            os.path.isfile(str(out_path) + "/ch_ways.csv") == False:
        way_check = 0
        ways_count = 0
        ways_dict = {}
        freespeed = None
        pbar = ProgressBar(widgets=[Bar('>', '[', ']'), ' ',
                                    Percentage(), ' ',
                                    ETA()], maxval=lines_links)

        # this reads WAYS file line by line, taking the information is relevant from attributes and children elements
        with gzip.open(str(out_path) + "\switzerland_network_ways.xml.gz") as f:
            # for line in tqdm_notebook(f, total=lines_ways):
            for line in pbar(f):
                if b"<link " in line:
                    #             records the way id and other attributes
                    way_check = 1
                    ways_count += 1
                    m = re.search(
                        rb'id="(.+)" from="(.+)" to="(.+)" length="(.+)" freespeed="(.+)" capacity="(.+)" permlanes="(.+)" oneway="(.+)" modes="(.+)"',
                        line)
                    if m:
                        way_id = m.group(1).decode('utf-8')
                        start_node_id = m.group(2).decode('utf-8')
                        end_node_id = m.group(3).decode('utf-8')
                        length = m.group(4).decode('utf-8')
                        freespeed = m.group(5).decode('utf-8')
                        capacity = m.group(6).decode('utf-8')
                        permlanes = m.group(7).decode('utf-8')
                        oneway = m.group(8).decode('utf-8')
                        modes = m.group(9).decode('utf-8')

                if way_check == 1:
                    m = re.findall(rb'<attribute name="osm:way:highway" class="java.lang.String" >(.*)</attribute>', line)
                    if m:
                        way_type = m[0].decode('utf-8')

                if b"</link>" in line:
                    # data of each way stored, saved in dictionary being the key the way_id
                    time = float(length) / float(freespeed)
                    way_data = [start_node_id,
                                end_node_id,
                                time,
                                length,
                                freespeed,
                                capacity,
                                permlanes,
                                oneway,
                                modes,
                                way_type]
                    if 'car' in modes:
                        ways_dict[way_id] = way_data

                    way_type = 'not_especify'
                    way_check = 0

        # EXPORT ways_dict TO FILE
        with open(str(out_path) + '/ch_ways_dict' + '.pkl', 'wb') as f:
            pickle.dump(ways_dict, f, pickle.HIGHEST_PROTOCOL)

        # EXPORT ways_europe TO FILE
        ch_ways_df = pd.DataFrame.from_dict(ways_dict, orient='index', columns=[
            "start_node_id", "end_node_id", "time", "length", "freespeed", "capacity", "permlanes", "oneway", "modes",
            "way_type"])
        ch_ways_df['way_id'] = ch_ways_df.index
        cols = ch_ways_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        ch_ways_df = ch_ways_df[cols]
        ch_ways_df.to_csv(str(out_path) + "/ch_ways.csv", sep=",", index=None)
        print('Number of ways in XML file: ' + str(ways_count))

    else:
        file = open(str(out_path) + "/ch_ways_dict.pkl", 'rb')
        ways_dict = pickle.load(file)
        ch_ways_df = pd.read_csv(str(out_path) + "/ch_ways.csv", sep=",")
        print('Ways dictionary already exists, loaded')

    print('N_ways in ways_dict and ch_ways_df (filter of "car" in modes applied): ' + str(len(ways_dict)))
    print('------------------------------------------------------------------------')

    # -------------------------------------------------------------------
    # CLEANING DUPLICATED DATA
    # -------------------------------------------------------------------
    if os.path.isfile(str(out_path) + "/clean_ways.csv") == False:
        train_o = ch_ways_df[['start_node_id', 'end_node_id']]
        train_d = pd.DataFrame.copy(train_o, deep=True)
        train_d2 = train_o.drop_duplicates().reset_index(drop=True)
        train_d.columns = ['end_node_id', 'start_node_id']
        # print(len(train_o))
        # print(len(train_d2))

        # MATCHING only in START AND END NODES
        # on this sample: 3969 rows have duplicates taking the start_node_id and end_node_id, this means there is different data for the same defined way in this cases
        # out of 3969: 3921 have 1 duplicate (first+1) and 48 have 2 duplicates (first+2)
        dup_ways_bynode = ch_ways_df[ch_ways_df.duplicated(['start_node_id', 'end_node_id'], keep=False)]
        # dup_ways_bynode= dup_ways_bynode.drop(['way_id'],axis = 1).drop_duplicates(keep=False) do not activate this
        dup_ways_bynode = dup_ways_bynode.sort_values(by=['start_node_id', 'end_node_id'])
        dup_ways_bynode = pd.merge(dup_ways_bynode, ch_ways_df, how='inner')
        dup_ways_bynode.to_csv(str(out_path) + "/dup_ways_bynode.csv", sep=",", index=None)
        print('There are ' + str(len(ch_ways_df[ch_ways_df.duplicated(['start_node_id','end_node_id'],keep='first')])) +
              ' links which have at least one duplicate.')

        # MATCHING IN DATA WITHOUT ID
        # on this sample: 450 rows have duplicates without counting the way_id(which is unique for all)
        # out of 450: 449 have 1 duplicate (first+1) and 1 has 2 duplicates (first+2)
        # no_id = ch_ways_df.drop(['way_id'], axis=1)
        # dup_id = no_id[no_id.duplicated(keep='first')].drop_duplicates(keep='first')
        # index of all first duplicates is saved in 'no_id_idx' for later filter to main df
        # no_id_idx = dup_id.index.values.tolist()
        # print(len(dup_id))

        # This selects the fastest way in cases where there are duplicates, for a later simpler graph
        idx_list = []

        j = -1
        actual_i = 0
        for i in range(0, len(dup_ways_bynode) - 1):
            if i > actual_i + j:
                actual_i = i
                start_node_id = str(dup_ways_bynode.iloc[i]['start_node_id'])
                end_node_id = str(dup_ways_bynode.iloc[i]['end_node_id'])

                times = []
                j = 0
                start_n = str(dup_ways_bynode.iloc[i + j]['start_node_id'])
                end_n = str(dup_ways_bynode.iloc[i + j]['end_node_id'])
                #         print(start_node_id,end_node_id,'o')
                while start_node_id == start_n and end_node_id == end_n:
                    #             print(start_n, end_n)
                    time = dup_ways_bynode.iloc[i + j]['time']
                    times.append(time)
                    j += 1
                    if i + j < len(dup_ways_bynode) - 1:
                        start_n = str(dup_ways_bynode.iloc[i + j]['start_node_id'])
                        end_n = str(dup_ways_bynode.iloc[i + j]['end_node_id'])
                    #                 print(start_n, end_n)
                    elif i + j == len(dup_ways_bynode) - 1:
                        j -= 1
                        time = dup_ways_bynode.iloc[i + j]['time']
                        times.append(time)
                        break
                j -= 1
                idx = times.index(min(times))
                idx_list.append(i + idx)

        # resulting dataframe after cleaning
        selected_ways = dup_ways_bynode.iloc[idx_list]
        unique_ways = ch_ways_df.drop_duplicates(['start_node_id', 'end_node_id'], keep=False)
        clean_ways = pd.concat([selected_ways, unique_ways], sort=False).sort_values(by=['way_id'])
        # print(len(selected_ways), len(dup_ways_bynode), len(unique_ways), len(ch_ways_df))
        clean_ways.to_csv(str(out_path) + "/clean_ways.csv", sep=",", index=None)
    else:
        clean_ways = pd.read_csv(str(out_path) + "/clean_ways.csv", sep=",")
        print('clean_ways already exists, loaded')
    print('Exported "clean_ways.csv" with fastest links in case of duplicates with ' + str(len(clean_ways)) + ' on it.')
    print('------------------------------------------------------------------------')

    # -------------------------------------------------------------------
    # CREATE GRAPH FROM NETWORK DATABASE
    # -------------------------------------------------------------------
    if os.path.isfile(str(out_path) + "/ch_MultiDiGraph_bytime.gpickle") == False or \
            os.path.isfile(str(out_path) + "/ch_DiGraph_bytime.gpickle") == False:
        # create MultiDiGraph to include all edges
        G = nx.MultiDiGraph()
        edges = ch_ways_df[["start_node_id", "end_node_id", "time", "way_id", "modes", "length"]]
        edges_list = edges.values.tolist()
        [G, G_isolated, isolated, largest] = create_graph(G,edges_list,out_path,'ch_MultiDiGraph_bytime',nodes_dict)

        # Also with 'clean_data' create a DiGraph for attributes comparison with fastest duplicated ways
        G_simple = nx.DiGraph()
        edges_s = clean_ways[["start_node_id", "end_node_id", "time", "way_id", "modes", "length"]]
        edges_list_s = edges_s.values.tolist()
        [G_simple, G_isolated, isolated, largest] = create_graph(G_simple, edges_list_s, out_path,
                                                                 'ch_DiGraph_bytime', nodes_dict)
    # LARGEST ISLAND OF GRAPH
        edges = G.edges(list(largest))
        create_shp(edges, 'ch_MultiDiGraph_bytime_largest')
        # ISOLATED GRAPH
        if len(list(isolated)) > 0:
            iso_edges = G_isolated.edges(list(isolated))
            create_shp(iso_edges, 'isolated_graph')
            # export GRAPH to file
            nx.write_gpickle(G_isolated, str(out_path) + "/isolated_graph.gpickle")
    else:
        print('G MultiDiGraph and DiGraph files already exists')

    print('------------------------------------------------------------------------')
    print('Process finished correctly: files created in out_path')
