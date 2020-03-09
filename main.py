# import sys
# sys.path.insert(0, 'codes')
import datetime
import os
import argparse

# functions from other scripts
# from network_graph import parse_network
from graph_analysis import filter_graph
from population_parser import population_parser_setup

print(datetime.datetime.now(), 'Main script begins ...')

# -------------------------------------------------------------------------------------------------------------
#  CREATE GRAPH FROM NETWORK XML FILE
# -------------------------------------------------------------------------------------------------------------
# '''
# Given the directory of the network XML file this function will: (avg time:15-20 mins)
# 1. Split them into different files for <nodes> and <links>
# 2. Create dictionaries from nodes and links containing respective information (where key = node/link id)
# 3. From the links database, create a MultiDiGraph removing any isolated networks on it
# 4. Create a shape file from the graph for visualization
# Created files in output directory:
# 1. Splitted XML files: switzerland_network_nodes.xml.gz/switzerland_network_ways.xml.gz
# 2. Dictionaries for nodes and links: ch_nodes_dict2056.pkl/ch_ways_dict.pkl (+ csv of links database)
# 3. Original graph/Largest connected graph/Isolated networks (for MultiDiGraph and DiGraph (with fastest links):
#       ch_MultiDiGraph_bytime.gpickle/ch_MultiDiGraph_bytime_largest.gpickle/isolated_graph.gpickle
#       ch_DiGraph_bytime.gpickle/ch_DiGraph_bytime_largest.gpickle/isolated_graph.gpickle
#       in case there are not isolated networks: the first 2 will be equal, and no third file will be created
# 4. Shape file of graph: ch_network_largest_graph_bytime.shp
# Needed arguments: 1. Directory with name to the network compressed XML file (i.e.: r"C:\Users\...\network.xml.gz")
#                   2. Output directory for all new files created (i.e.: r"C:\Users\...\network")
# '''
# parse_network(r"C:\Users\Ion\TFM\data\scenarios\switzerland_1pm\switzerland_network.xml.gz",
#               r"C:\Users\Ion\TFM\data\network_graphs\test")

# -------------------------------------------------------------------------------------------------------------
# PARSE AND CREATE POPULATION DATABASE FROM XML FILE
# -------------------------------------------------------------------------------------------------------------
# '''
# Population filtering by an XML parser:
# This function returns in the output directory both, attributes and plans dictionaries and csv files
# of homogeneus data
# 1. Attributes: normal population is defined by 10 attributes, freight with 3 attributes
# 2. Plans: every tuple means every trip, with information of origin and destination points and the transport
# Needed arguments: 1. Directory containing the compressed scenario file population file (i.e.: r"C:\Users\...\data\scenarios\")
#                   2. Output directory to save new files
#                   3. Scenario (i.e. 'switzerland_1pm', 'switzerland_1pct', 'switzerland_10pct'
# '''
# population_parser_line(scenarios_dir)
#
# pop_list = ['switzerland_1pm', 'switzerland_1pct', 'switzerland_10pct']
# for scenario in pop_list:
#     area = None
#     population_parser_line(r'C:\Users\Ion\TFM\data\scenarios',
#                            r'C:/Users/Ion/TFM/data/study_areas',
#                            scenario,
#                            area)
#
#
# area_list = [dI for dI in os.listdir(r'C:/Users/Ion/TFM/data/study_areas') if
#              os.path.isdir(os.path.join(r'C:/Users/Ion/TFM/data/study_areas', dI))]
# for area in area_list:
#     scenario = None
#     population_parser_line(r'C:/Users/Ion/TFM/data/scenarios',
#                            r'C:/Users/Ion/TFM/data/study_areas',
#                            scenario,
#                            area)

# population_parser_setup(r'C:/Users/Ion/TFM/data/scenarios')
# population_parser_setup(r'C:/Users/Ion/TFM/data/study_areas')

# for area in area_list:
#     population_parser_line(r'C:/Users/Ion/TFM/data/study_areas',
#                            area)
#     population_parser_line(r'C:/Users/Ion/TFM/data/study_areas/' + str(area) + '/' + str(area) + '_population.xml.gz',
#                            r'C:/Users/Ion/TFM/data/study_areas/' + str(area) + '/population_db',
#                            r'C:/Users/Ion/TFM/data/study_areas')


# -------------------------------------------------------------------------------------------------------------
# CREATE AND ANALYSE EVERY STUDY AREA
# -------------------------------------------------------------------------------------------------------------
# '''
# From the created graphs (ch_MultiDiGraph_bytime_largest.gpickle/ch_DiGraph_bytime_largest.gpickle)
# and given different study areas in shape files stored in the same directory
# but each on a different folder (for the output files)
# Needed arguments: 1. Directory containing the folders with the study areas (i.e.: r"C:\Users\...\network")
#                   2. Directory of out_path from create_graph() where graph and nodes_dict files are
#                   (i.e.: r"C:\Users\...\network\graph.gpickle")
# '''

filter_graph(r"C:/Users/Ion/TFM/data/study_areas",
             r"C:/Users/Ion/TFM/data/network_graphs",
             'test_area')

# parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
# parser.add_argument('--study-areas', dest="study_areas", help='path to study areas')
# parser.add_argument('--network-graphs', dest="network_graphs", help="path to network_graphs")
# args = parser.parse_args()
# filter_graph(args.study_areas, args.network_graphs)

# -------------------------------------------------------------------------------------------------------------
print(datetime.datetime.now(), 'Main script closing ...')