import matplotlib.pyplot as plt
import pandas as pd
x= [1,2,3,4,5,6,7,8,9]
y= [0.1,0.2,0.3,0.5,0.6,0.65,0.8,1.3,1.5]
plt.plot(x,y)

# change table attributes, to avg values
study_area_dir= r"C:\Users\Ion\TFM\data\study_areas"
attr_df = pd.read_csv(str(study_area_dir) + '/' + 'attribute_table.csv', sep=",", index_col='attributes', dtype=object)

# this takes from every cell (if it is a list) the value in 4th position which corresponds to the avg value
def take_avg(attr_df):
    # in case important columns are converted to str, here are back to float
    index_list = ['n_nodes', 'n_edges', 'network_distance', 'area']
    for column in attr_df.columns:
        for index in index_list:
            variable = float(attr_df.loc[index, column])
            attr_df.at[index, column] = variable

    # reindex dataframe
    new_index = ['n_nodes', 'n_edges', 'network_distance', 'area', 'avg_degree', 'degree_centrality',
                 'avg_degree_connectivity', 'avg_edge_density',
                 'avg_shortest_path_duration', 'node_betweenness*', 'edge_betweenness',
                 'node_load_centrality', 'edge_load_centrality', 'clustering*',
                 'eccentricity', 'radius', 'diameter', 'center_nodes', 'periphery_nodes',
                 'barycenter_nodes', 'node_closeness*', 'avg_neighbor_degree',
                 'node_straightness', 'clustering_w*']
    attr_df = attr_df.reindex(new_index)

    # sort both dataframes
    attr_df = attr_df.sort_values(by='network_distance', ascending=False, axis=1)

    # take avg of cells (if possible)
    attr_df_avg = attr_df.copy(deep=True)
    for i in range(len(attr_df)):
        # print(attr_df.index[i], i)
        for column in attr_df.columns:
            try:
                value = float(attr_df.iloc[i][column])
            except:
                ini_list = attr_df.iloc[i][column]
                res = ini_list.strip('][').split(', ')
                if attr_df.index[i] in ['center_nodes', 'periphery_nodes', 'barycenter_nodes']:
                    value = ini_list
                    attr_df_avg.at[attr_df.index[i], column] = value
                    continue
                value = res[3]
            attr_df_avg.at[attr_df.index[i], column] = value

    # export both df to csv
    attr_df.to_csv(str(study_area_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
    attr_df_avg.to_csv(str(study_area_dir) + "/attribute_table_AVG.csv", sep=",", index=True, index_label=['attributes'])
    return attr_df, attr_df_avg


attr_df, attr_df_avg = take_avg(attr_df)


