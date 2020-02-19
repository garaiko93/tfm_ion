import pandas as pd
import ast

# functions from other scripts
from graph_analysis import save_attr_df

# change table attributes, to avg values
# study_area_dir= r"C:\Users\Ion\TFM\data\study_areas"
# attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)
#
# def take_avg(attr_df, study_area_dir):
#     # take avg of cells (if possible)
#     attr_avg_df = attr_df.copy(deep=True)
#
#     for i in range(len(attr_df)):
#         # print(attr_df.index[i], i)
#         for column in attr_df.columns:
#             try:
#                 # INTEGER ATTRIBUTES
#                 if attr_df.index[i] in ['n_nodes', 'n_edges', 'population', 'trips',
#                                         'radius', 'diameter', 'n_intersection', 'n_street']:
#                     str_val = attr_df.iloc[i][column]
#                     int_val = int(float(str_val))
#                     attr_df.at[attr_df.index[i], column] = int_val
#                     attr_avg_df.at[attr_avg_df.index[i], column] = int_val
#
#                 # FLOAT ATTRIBUTES
#                 elif attr_df.index[i] in ['network_distance', 'area', 'avg_degree', 'avg_edge_density',
#                                           'avg_shortest_path_duration', 'streets_per_node', 'node_d_km',
#                                           'intersection_d_km', 'edge_d_km', 'street_d_km', 'circuity_avg']:
#                     str_val = attr_df.iloc[i][column]
#                     flo_val = float(str_val)
#                     attr_df.at[attr_df.index[i], column] = flo_val
#                     attr_avg_df.at[attr_avg_df.index[i], column] = flo_val
#
#                 # LIST ATTRIBUTES: [n, min, max, avg]
#                 elif attr_df.index[i] in ['degree_centrality', 'avg_degree_connectivity', 'node_betweenness*',
#                                           'edge_betweenness', 'node_load_centrality', 'edge_load_centrality',
#                                           'clustering*', 'eccentricity', 'node_closeness*', 'avg_neighbor_degree',
#                                           'node_straightness', 'clustering_w*']:
#                     str_val = attr_df.iloc[i][column]
#                     list_val = ast.literal_eval(str_val)
#                     attr_df.at[attr_df.index[i], column] = list_val
#                     attr_avg_df.at[attr_avg_df.index[i], column] = list_val[3]
#
#                 # elif attr_df.index[i] in ['center_nodes', 'periphery_nodes', 'barycenter_nodes']:
#                     # str_val = attr_df.iloc[i][column]
#                     # flo_val = int(str_val[2:-2])
#                     # attr_avg_df.drop(attr_avg_df.index[i])
#                     # attr_df_avg.at[attr_df_avg.index[i], column] = flo_val
#             except:
#                 attr_df.at[attr_df.index[i], column] = None
#                 attr_avg_df.at[attr_avg_df.index[i], column] = None
#
#
#     # reindex dataframe
#     new_index = ['n_nodes',            #stats_basic
#                   'n_edges',            #stats_basic
#                   'network_distance',   #stats_basic
#                   'area',
#                   'population',
#                   'trips',
#                   'n_intersection',     #stats_basic
#                   'n_street',           #stats_basic
#                   'streets_per_node',   #stats_basic
#                   'node_d_km',          #stats_basic
#                   'intersection_d_km',  #stats_basic
#                   'edge_d_km',     #stats_basic
#                   'street_d_km',    #stats_basic
#                   'circuity_avg',    #stats_basic
#                   'avg_degree',
#                   'avg_neighbor_degree',
#                   'degree_centrality',
#                'avg_degree_connectivity',
#                'avg_edge_density',
#                'avg_shortest_path_duration',
#                'node_betweenness*',
#                'edge_betweenness',
#                 'node_straightness',
#                'node_closeness*',
#                'node_load_centrality',
#                'edge_load_centrality',
#                   'clustering*',
#                   'clustering_w*',
#                'eccentricity',
#                'radius',
#                'diameter',
#                'center_nodes',
#                'periphery_nodes',
#                'barycenter_nodes']
#     attr_df = attr_df.reindex(new_index)
#     attr_avg_df = attr_avg_df.reindex(new_index)
#
#     # sort both dataframes
#     attr_df = attr_df.sort_values(by='network_distance', ascending=False, axis=1)
#     attr_avg_df = attr_avg_df.sort_values(by='network_distance', ascending=False, axis=1)
#
#     # delete some columns and rows in avg table
#     attr_avg_df = attr_avg_df.drop(index=['center_nodes', 'periphery_nodes', 'barycenter_nodes'],
#                                    columns=['bern_large', 'zurich_large', 'lausanne_lake'])
#
#     # create a transpose of the matrix
#     attr_avg_dfT = attr_avg_df.transpose()
#
#     # save the three dataframes to csv
#     save_attr_df(attr_df, study_area_dir, attr_avg_df, attr_avg_dfT)
#     # return attr_df, attr_avg_df


# this takes from every cell (if it is a list) the value in 4th position which corresponds to the avg value
# def take_avg(attr_df, study_area_dir):
#     # in case important columns are converted to str, here are back to float
#     index_list = ['n_nodes', 'n_edges', 'network_distance', 'area']
#     for column in attr_df.columns:
#         for index in index_list:
#             variable = float(attr_df.loc[index, column])
#             attr_df.at[index, column] = variable
#
#     # reindex dataframe
#     new_index = ['n_nodes', 'n_edges', 'network_distance', 'area', 'population', 'trips',
#                  'avg_degree', 'degree_centrality',
#                  'avg_degree_connectivity', 'avg_edge_density',
#                  'avg_shortest_path_duration', 'node_betweenness*', 'edge_betweenness',
#                  'node_load_centrality', 'edge_load_centrality', 'clustering*',
#                  'eccentricity', 'radius', 'diameter', 'center_nodes', 'periphery_nodes',
#                  'barycenter_nodes', 'node_closeness*', 'avg_neighbor_degree',
#                  'node_straightness', 'clustering_w*']
#     attr_df = attr_df.reindex(new_index)
#
#     # sort both dataframes
#     attr_df = attr_df.sort_values(by='network_distance', ascending=False, axis=1)
#
#     # take avg of cells (if possible)
#     attr_df_avg = attr_df.copy(deep=True)
#     for i in range(len(attr_df)):
#         # print(attr_df.index[i], i)
#         for column in attr_df.columns:
#             try:
#                 value = float(attr_df.iloc[i][column])
#             except:
#                 ini_list = attr_df.iloc[i][column]
#                 res = ini_list.strip('][').split(', ')
#                 if attr_df.index[i] in ['center_nodes', 'periphery_nodes', 'barycenter_nodes']:
#                     value = ini_list
#                     attr_df_avg.at[attr_df.index[i], column] = value
#                     continue
#                 value = res[3]
#             attr_df_avg.at[attr_df.index[i], column] = value
#
#     # export both df to csv
#     attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
#     attr_df_avg.to_csv(str(study_area_dir) + "/attribute_table_AVG.csv", sep=",", index=True, index_label=['attributes'])
#     return attr_df, attr_df_avg


# attr_df, attr_df_avg = take_avg(attr_df)


