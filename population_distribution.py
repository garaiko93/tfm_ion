import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import math
import pandas as pd
import copy
import itertools
import ntpath
import datetime

def create_distrib(area_path, grid_size, create_map=True):
    print(datetime.datetime.now(), 'Computing facility distribution and exporting as dataframe for every grid area ...')
    area = ntpath.split(area_path)[1]
    study_area_dir = ntpath.split(area_path)[0]

    study_area_shp = gpd.read_file(str(area_path) + "/" + str(area) + ".shp").iloc[0]['geometry']
    area_pop = pd.read_csv(str(area_path) + "/population_db/loc_home.csv")
    area_empl = pd.read_csv(str(area_path) + "/population_db/loc_work.csv")
    area_shop = pd.read_csv(str(area_path) + "/population_db/loc_shop.csv")
    area_edu = pd.read_csv(str(area_path) + "/population_db/loc_education.csv")
    area_leisure = pd.read_csv(str(area_path) + "/population_db/loc_leisure.csv")

    # drop_rows = []
    # for index, row in fac_df.iterrows():
    #     point = Point(row['x'], row['y'])
    #     in_area = study_area_shp.contains(point)
    #     if not in_area:
    #         drop_rows.append(index)
    # area_fac = fac_df.drop(drop_rows, axis=0)
    # print(len(fac_df), len(area_fac))

    # Found borders of the grid in 4 cardinal points based on the shape file
    cr = list(study_area_shp.exterior.coords)

    xs = []
    ys = []
    for x, y in cr:
        xs.append(x)
        ys.append(y)

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)


    # min_x = area_fac['x'].min()
    # min_y = area_fac['y'].min()
    #
    # max_x = area_fac['x'].max()
    # max_y = area_fac['y'].max()

    # define origin of coordinates top-corner
    o_x = min_x - 50
    o_y = max_y + 50
    o = Point(o_x, o_y)

    # dimensions of grid
    x_axis = max_x - o_x
    y_axis = o_y - min_y

    x = math.ceil(x_axis/grid_size) #rounds up the number
    y = abs(math.ceil(y_axis/grid_size)) #rounds up the number
    m = [np.zeros([y, x]), np.zeros([y, x])]
    print(datetime.datetime.now(), 'Number of grids composing the grid: ' + str(y*x))

    def create_m(m, fac_list, grid_size, o_x, o_y):
        m_fac = copy.deepcopy(m)
        for area_fac in fac_list:
            for index, row in area_fac.iterrows():
                x_coord = row['x'] - o_x
                m_x = math.floor(x_coord / grid_size)

                y_coord = o_y - row['y']
                m_y = math.floor(y_coord / grid_size)

                m_fac[0][(m_y, m_x)] += row['n_persons']
        return m_fac

    m_pop = create_m(m, [area_pop], grid_size, o_x, o_y)
    m_empl = create_m(m, [area_edu, area_empl], grid_size, o_x, o_y)
    m_opt = create_m(m, [area_leisure, area_shop], grid_size, o_x, o_y)

    # create geodataframe with each square from matrix m
    id = itertools.count()
    m_df = pd.DataFrame(data=None, columns=['id', 'geometry'])
    for i in range(len(m[0])):
        for j in range(len(m[0][0])):
            points = [((o_x + (grid_size * j)), (o_y - (grid_size * i))),
                      ((o_x + (grid_size * (j + 1))), (o_y - (grid_size * i))),
                      ((o_x + (grid_size * (j + 1))), (o_y - (grid_size * (i + 1)))),
                      ((o_x + (grid_size * j)), (o_y - (grid_size * (i + 1))))]
            polygon = Polygon(points)

            # for the grid squares in the border of study area, for a less than 30% of coincidence, it is avoid
            # for larger than 30% coincidence, proportional density over a full coincidence square is assigned
            if polygon.intersects(study_area_shp):
                percentage = study_area_shp.intersection(polygon).area / (grid_size ** 2)
                if percentage > 0.3:
                    pop_value = m_pop[0][i][j] / percentage
                    empl_value = m_empl[0][i][j] / percentage
                    opt_value = m_opt[0][i][j] / percentage
                else:
                    continue
            else:
                pop_value = m_pop[0][i][j]
                empl_value = m_empl[0][i][j]
                opt_value = m_opt[0][i][j]

            if m_pop[0][i][j] == 0 and m_empl[0][i][j] == 0 and m_opt[0][i][j] == 0:
                continue

            poly_2056 = Polygon(points)
            centroid = poly_2056.centroid

            new_row = {'id': str(next(id)),
                       'pop': pop_value,
                       'empl': empl_value,
                       'opt': opt_value,
                       'geometry': poly_2056,
                       'centroid': [centroid.x, centroid.y]
                       }
            m_df = m_df.append(new_row, ignore_index=True)
    if create_map:
        m_gdf = gpd.GeoDataFrame(m_df.loc[:, m_df.columns != 'centroid'])
        m_gdf.to_file(str(area_path) + "/" + str(area) + '_' + "facility_distribution_" +
                      str(grid_size) + "gs.shp")

    # return m_df.loc[:, m_df.columns != 'geometry']
    return m_df


# area_path = r"C:/Users/Ion/TFM/data/study_areas/zurich_kreis"
# m_df = create_distrib(area_path, 1000, True)


# for area_fac in fac_list:
#     c = itertools.count()
#     counter = next(c)
#     for index, row in area_fac.iterrows():
#         x_coord = row['x'] - o_x
#         m_x = math.floor(x_coord / grid_size)
#
#         y_coord = o_y - row['y']
#         m_y = math.floor(y_coord / grid_size)
#
#         m[counter][(m_y, m_x)] += row['n_persons']
#
#     if create_map:
#         # create geodataframe with each square from matrix m
#         c = itertools.count()
#         id = 0
#         m_df = pd.DataFrame(data=None, columns=['id', 'geometry'])
#         for i in range(len(m[counter])):
#             for j in range(len(m[counter][0])):
#                 points_4326 = []
#                 points = [((o_x + (grid_size * j)), (o_y - (grid_size * i))),
#                           ((o_x + (grid_size * (j+1))), (o_y - (grid_size * i))),
#                           ((o_x + (grid_size * (j+1))), (o_y - (grid_size * (i+1)))),
#                           ((o_x + (grid_size * j)), (o_y - (grid_size * (i+1))))]
#                 polygon = Polygon(points)
#                 # for the grid squares in the border of study area, for a less than 30% of coincidence, it is avoid
#                 # for larger than 30% coincidence, proportional density over a full coincidence square is assigned
#                 if polygon.intersects(study_area_shp):
#                     percentage = study_area_shp.intersection(polygon).area / (grid_size ** 2)
#                     m[len(fac_list)][i][j] = percentage
#                     if percentage > 0.3:
#                         value = m[0][i][j] / percentage
#                     else:
#                         continue
#                 else:
#                     m[len(fac_list)][i][j] = 0 #its len(fac_list) to ensure the intersection % is in the last layer of m
#
#                 if m[counter][i][j] == 0: continue

        # transform coordinates system from epsg 2056 to epsg 4326
        # transformer = Transformer.from_crs(2056, 4326)
        # for pt in transformer.itransform(points):
        #     points_4326.append((pt[1],pt[0]))
        # poly_4326 = Polygon(points_4326)
#         poly_2056 = Polygon(points)
#         # value = m[0][i][j]
#         new_row = {'id': str(id),
#                    'value': value,
#                    'geometry': poly_2056
#                    }
#         m_df = m_df.append(new_row, ignore_index=True)
#         id += 1
#
# m_gdf = gpd.GeoDataFrame(m_df)
# m_gdf.to_file(r"C:/Users/Ion/TFM/data/study_areas/zurich_kreis/" + str(area) + "pop_distribution_" +
#               str(grid_size) + "gs.shp")
# m_gdf.to_file(r"C:\Users\Ion\TFM\data\study_areas\zurich_kreis/output.json", driver="GeoJSON")

# create shp file with points in area
# point_list = []
# for index,row in area_fac.iterrows():
#     point = Point(row['x'], row['y'])
#     point_list.append(point)
# area_fac['geometry'] = point_list
#
# area_gdf = gpd.GeoDataFrame(area_fac)
# area_gdf.to_file(r"C:\Users\Ion\TFM\data\study_areas\zurich_small/home_points.shp")

# Initialize the map:
# map_name = str(area) + '_' + str(pct) + '_' + str(grid_size) + 'gs'
# map2 = folium.Map(location=[47.376155, 8.531508], zoom_start=12)
# # Add the color for the chloropleth:
# folium.Choropleth(
#     geo_data=r"C:\Users\Ion\TFM\data\study_areas\zurich_kreis/output.json",
#     # geo_data=m_gdf[['id', 'geometry']],
#     name='choropleth',
#     data=m_df,
#     columns=['id', 'value'],
#     key_on='feature.properties.id',
#     fill_color='YlOrRd',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name=map_name
# ).add_to(map2)
# # Save to html
# # map2.save(os.path.join(r"C:\Users\Ion\TFM\data\study_areas\zurich_small/", str(map_name) + '.html'))
# map2.save(os.path.join(r"C:\Users\Ion\TFM\data\study_areas\zurich_kreis/", str(map_name) + '1.shp'))
# return m_pop, m_empl, m_opt



# study_area_gdf.iloc[0]['geometry']
# code finishes here
# ------------------------------------------
# OTHER WAYS TO PLOT THE MAP IN THE BACKGROUND
# Create a dataset (fake)
#     df = pd.DataFrame(m[1])
#
#     # plot using a color palette
#     sns.heatmap(df, cmap="YlGnBu")
#
#     # add this after your favorite color to show the plot
#     # sns.plt.show(block=True)
#
#     # left
#     sns.heatmap(df, vmin=0, vmax=0.5)
#     sns.plt.show()
#     # #right
#     # sns.heatmap(df, vmin=0.5, vmax=0.7)
#     # sns.plt.show()
#     poly_list = []
#     for index, row in df.iterrows(): # in range(0,len(df)-1):
#         poly = Polygon([(o_x, (o_y - (grid_size*(index)))),
#                         (max_x, (o_y - (grid_size*(index)))),
#                         (max_x, (o_y - (grid_size*(index+1)))),
#                         (o_x, (o_y - (grid_size*(index+1))))])
#         poly_list.append(poly)
#
#     df['geometry'] = poly_list
#     df = gpd.GeoDataFrame(df)
#
#     df = gpd.read_file(r'C:\Users\Ion\TFM\data\study_areas\zurich_small\data/stzh.adm_stadtkreise_v_polygon.shp')
#     ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
#
#
#     df.crs = {'init': 'epsg:2056'}
#     df = df.to_crs(epsg=3857)
#     ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
#     ctx.add_basemap(ax)

# how to combine polygons from a gdf and safe to file the final polygon
# poly_list = []
# for index, row in study_area_gdf.iterrows():
#     poly_list.append(row['geometry'])
# study_area_shp = cascaded_union(poly_list)
# m_ggdf = pd.DataFrame(data=None, columns=['id', 'geometry'])
# new_row = {'id': 0,
#            'geometry': study_area_shp
#            }
# m_ggdf = m_ggdf.append(new_row, ignore_index=True)
# area_gdf = gpd.GeoDataFrame(m_ggdf)
# area_gdf.to_file(r"C:\Users\Ion\TFM\data\study_areas\zurich_small/zurich_kreis.shp")