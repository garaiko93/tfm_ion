import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import math
import pandas as pd
import folium
import os
from pyproj import Transformer
from shapely.ops import cascaded_union

def create_distrib(study_area_shp, fac_df, grid_size):
    drop_rows = []
    for index, row in fac_df.iterrows():
        point = Point(row['x'], row['y'])
        in_area = study_area_shp.contains(point)
        if not in_area:
            drop_rows.append(index)
    area_fac = fac_df.drop(drop_rows, axis=0)
    print(len(fac_df), len(area_fac))

    # Found borders of the grid in 4 cardinal points
    min_x = area_fac['x'].min()
    min_y = area_fac['y'].min()

    max_x = area_fac['x'].max()
    max_y = area_fac['y'].max()

    # define origin of coordinates top-corner
    o_x = min_x - 50
    o_y = max_y + 50
    o = Point(o_x, o_y)

    # dimensions of grid
    x_axis = max_x - o_x
    y_axis = o_y - min_y

    x = math.ceil(x_axis/grid_size) #rounds up the number
    y = abs(math.ceil(y_axis/grid_size)) #rounds up the number
    m = [np.zeros([y,x]),np.zeros([y,x])]
    print(y*x)

    for index, row in area_fac.iterrows():
        x_coord = row['x'] - o_x
        m_x = math.floor(x_coord / grid_size)

        y_coord = o_y - row['y']
        m_y = math.floor(y_coord / grid_size)

        m[0][(m_y,m_x)] += row['n_persons']

    # create geodataframe with each square from matrix m
    id = 0
    m_df = pd.DataFrame(data=None, columns=['id', 'geometry'])
    for i in range(0, len(m[0])):
        for j in range(0, len(m[0][0])):
            points_4326 = []
            points = [((o_x + (grid_size * j)), (o_y - (grid_size * i))),
                      ((o_x + (grid_size * (j+1))), (o_y - (grid_size * i))),
                      ((o_x + (grid_size * (j+1))), (o_y - (grid_size * (i+1)))),
                      ((o_x + (grid_size * j)), (o_y - (grid_size * (i+1))))]
            polygon = Polygon(points)
            # for the grid squares in the border of study area, for a less than 30% of coincidence, it is avoid
            # for larger than 30% coincidence, proportional density over a full coincidence square is assigned
            if polygon.intersects(study_area_shp):
                percentage = study_area_shp.intersection(polygon).area / (grid_size ** 2)
                m[1][i][j] = percentage
                if percentage > 0.3:
                    value = m[0][i][j] / percentage
                else:
                    continue
            else:
                m[1][i][j] = 0

            if m[0][i][j] == 0: continue

            # transform coordinates system from epsg 2056 to epsg 4326
            transformer = Transformer.from_crs(2056, 4326)
            for pt in transformer.itransform(points):
                points_4326.append((pt[1],pt[0]))
            poly_4326 = Polygon(points_4326)
            # value = m[0][i][j]
            new_row = {'id': str(id),
                       'value': value,
                       'geometry': poly_4326
                       }
            m_df = m_df.append(new_row, ignore_index=True)
            id += 1

    m_gdf = gpd.GeoDataFrame(m_df)
    m_gdf.to_file(r"C:\Users\Ion\TFM\data\study_areas\zurich_small/output.json", driver="GeoJSON")

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
    map_name = str(area) + '_' + str(pct) + '_' + str(grid_size) + 'gs'
    map2 = folium.Map(location=[47.376155, 8.531508], zoom_start=12)
    # Add the color for the chloropleth:
    folium.Choropleth(
        geo_data=r"C:\Users\Ion\TFM\data\study_areas\zurich_small/output.json",
        # geo_data=m_gdf[['id', 'geometry']],
        name='choropleth',
        data=m_df,
        columns=['id', 'value'],
        key_on='feature.properties.id',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=map_name
    ).add_to(map2)
    # Save to html
    map2.save(os.path.join(r"C:\Users\Ion\TFM\data\study_areas\zurich_small/", str(map_name) + '.html'))
    return m

area = "zurich_small"
facility = "home"
pct = "1pct"
population_path = r"C:/Users/Ion/TFM/data/population_db/test/switzerland_" + str(pct)
shp_path = r"C:/Users/Ion/TFM/data/study_areas" + "/" + str(area)
study_area_gdf = gpd.read_file(str(shp_path) + "/" + area + ".shp")
study_area_gdf = gpd.read_file(str(shp_path) + "/" + area + ".shp").iloc[1]['geometry']
fac_df = pd.read_csv(str(population_path) + "/loc_" + str(facility) + ".csv")
# grid_size = int(input('Enter grid size in meters: '))
grid_size = 1000
m = create_distrib(study_area_shp, fac_df, grid_size)


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