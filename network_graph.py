import gzip
import re
import pandas as pd
import pickle
import geopandas as gpd
import shapely.geometry as geo
import networkx as nx
import copy
import os
from shutil import copyfile

# Identify the largest component and the "isolated" nodes
def check_iso_graph(G,out_path):
    print('Initial graph has: ' + str(
        len([len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)])) + ' island with '
          + str(G.number_of_nodes()) + ' nodes and ' + str(G.number_of_edges()) + ' edges.')
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
        nx.write_gpickle(G, str(out_path) + "\ch_network_largest_graph_bytime.gpickle")
    else:
        largest = components[0]
        num_isolated = 0
        isolated = []
        longest_networks = len(components[0])
        copyfile(str(out_path) + "\ch_network_graph_bytime.gpickle",
                 str(out_path) + "\ch_network_largest_graph_bytime.gpickle")
    print('N isolated nodes: ' + str(num_isolated))
    print('10 largest networks (nodes): ' + str(longest_networks))
    print('Final graph has: ' + str(
        len([len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)])) + ' island with '
          + str(G.number_of_nodes()) + ' nodes and ' + str(G.number_of_edges()) + ' edges.')
    return G,isolated, largest


def create_graph(raw_file, out_path):
    if not os.path.exists(str(out_path)):
        os.makedirs(str(out_path))
        print('Directory created')
    else:
        print('Directory exists')

    lines_nodes = 881661
    lines_ways = 7181593
    lines_europe = 8063269
    # #SPLIT OSM FILE IN FILES FOR: NODES, WAYS
    node_check = 0
    way_check = 0
    nodes_n = 0
    ways_n = 0
    lines_nodes = 0
    lines_ways = 0
    lines_europe = 0

    # by reading the selected file line by line, for each xml element type this code splits the 3 of them in 3 different files for: NODES, WAYS and RELATIONS elements
    with gzip.open(raw_file) as f:
        with gzip.open(str(out_path)+"\switzerland_network_nodes.xml.gz", 'wb') as f1:
            with gzip.open(str(out_path)+"\switzerland_network_ways.xml.gz", 'wb') as f2:
                for line in (f):
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
                        lines_ways += 1
                        if b"</link>" in line:
                            way_check = 0

    # SCREEN PRINT
    # print(nodes_n)
    # print(ways_n)
    print(lines_nodes)
    print(lines_ways)
    print(lines_europe)
    print('Raw file splitted correctly in out_path')
    # -----------------------------------------------------------------------------
    # NODES
    # -----------------------------------------------------------------------------
    nodes_dict = {}
    with gzip.open(str(out_path) + "\switzerland_network_nodes.xml.gz") as f:
        #     reading line by line the 'nodes' file created at the beginning, data for each node fulfilling the conditions are stored for the output
        # for line in tqdm(f, total=lines_nodes):
        for line in f:
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
    with open(str(out_path) + '\ch_nodes_dict2056' + '.pkl', 'wb') as f:
        pickle.dump(nodes_dict, f, pickle.HIGHEST_PROTOCOL)
    # with open(out_path / 'ch_nodes_dict2056.pkl', 'wb') as f:
    #     pickle.dump(nodes_dict, f, pickle.HIGHEST_PROTOCOL)
    print('Nnodes in nodes_dict: ' + str(len(nodes_dict)))

    # WAYS
    way_check = 0
    ways_count = 0
    ways_dict = {}
    freespeed = None

    # this reads WAYS file line by line, taking the information is relevant from attributes and children elements
    with gzip.open(str(out_path) + "\switzerland_network_ways.xml.gz") as f:
        # for line in tqdm_notebook(f, total=lines_ways):
        for line in f:
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
    with open(str(out_path) + '\ch_ways_dict' + '.pkl', 'wb') as f:
        pickle.dump(ways_dict, f, pickle.HIGHEST_PROTOCOL)

    # EXPORT ways_europe TO FILE
    ch_ways_df = pd.DataFrame.from_dict(ways_dict, orient='index', columns=[
        "start_node_id", "end_node_id", "time", "length", "freespeed", "capacity", "permlanes", "oneway", "modes",
        "way_type"])
    ch_ways_df['way_id'] = ch_ways_df.index
    cols = ch_ways_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    ch_ways_df = ch_ways_df[cols]
    ch_ways_df.to_csv(str(out_path) + "\ch_ways.csv", sep=",", index=None)

    print('Nways in ways_dict: ' + str(len(ways_dict)))
    print(ways_count)
    print('Nways in ch_ways_df: ' + str(len(ch_ways_df)))

    # -------------------------------------------------------------------
    # Create a graph from network
    # -------------------------------------------------------------------
    G = nx.MultiDiGraph()

    # import the network created from the OSM file
    # edges = pd.read_csv(str(out_path)+"\ch_ways.csv") [["start_node_id", "end_node_id", "time", "way_id","modes"]]
    edges = ch_ways_df[["start_node_id", "end_node_id", "time", "way_id", "modes"]]
    edges_list = edges.values.tolist()

    # introduce every way as edge with attributes of time and new_id
    for start, end, time, way_id, modes in edges_list:
        G.add_edge(start, end, time=time, way_id=way_id, modes=modes)
    start_Nn = G.number_of_nodes()
    start_Ne = G.number_of_edges()

    # add attributes of coordinates to each node
    for node in list(G.nodes):
        G.nodes[node]['x'] = nodes_dict[node][0]
        G.nodes[node]['y'] = nodes_dict[node][1]
    # export graph of original network to file (without excluding any edge or island)
    nx.write_gpickle(G, str(out_path) + "\ch_network_graph_bytime.gpickle")
    G_isolated = copy.deepcopy(G)
    print('Original graph has: ' + str(
        len([len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)])) + ' island with '
          + str(G.number_of_nodes()) + '/' + str(G.number_of_edges()) + ' (Nnodes/Nedges)')

    [G, isolated, largest] = check_iso_graph(G,out_path)

    end_Nn = G.number_of_nodes()
    end_Ne = G.number_of_edges()

    print('Input edges: ' + str(len(edges_list)))
    print('Start/End N_nodes: ' + str(start_Nn) + '/' + str(end_Nn))
    print('Start/End N_edges: ' + str(start_Ne) + '/' + str(end_Ne))

    print('Resulting graph has: ' + str(
        len([len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)])) + ' island with '
          + str(G.number_of_nodes()) + '/' + str(G.number_of_edges()) + ' (Nnodes/Nedges)')


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

    # LARGEST ISLAND OF GRAPH
    edges = G.edges(list(largest))
    create_shp(edges, 'ch_network_largest_graph_bytime')
    print(len(list(edges)))
    # ISOLATED GRAPH
    if len(list(isolated)) > 0:
        iso_edges = G_isolated.edges(list(isolated))
        create_shp(iso_edges, 'isolated_graph')
        # export GRAPH to file
        nx.write_gpickle(G_isolated, str(out_path) + "\isolated_graph.gpickle")
        print(len(list(iso_edges)))
    print('Process finished correctly: graph created in out_path')