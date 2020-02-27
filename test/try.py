import pandas as pd
import os
import gzip

trips =pd.read_csv('C:/User')

file = open(r'C:\Users\Ion\TFM\data\study_areas\sion/attr_node_betweenness.pkl', 'rb')
nodes_dict = pickle.load(file)
dict_data(nodes_dict, r'C:\Users\Ion\TFM\data\study_areas\sion', 'node_straightness')


for subdir, dirs, files in os.walk(r'C:\Users\Ion\TFM\data\study_areas'):
    for file in files:
        print (os.path.join(subdir, file))

for subdir in os.walk(r'C:\Users\Ion\TFM\data\study_areas'):
    print (subdir)
foo = r'C:\Users\Ion\TFM\data\study_areas'
output = [dI for dI in os.listdir(foo) if os.path.isdir(os.path.join(foo,dI))]

pop_file = r'C:\Users\Ion\TFM\data\study_areas/zurich_kreis/zurich_kreis_population.xml.gz'
count = 0
with gzip.open(pop_file) as f:
    #     reading line by line the 'nodes' file created at the beginning, data for each node fulfilling the conditions are stored for the output
    for line in f:
        if b"<person " in line:
            count += 1
print(count)

from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import geopandas as gpd
import pandas as pd
study_area_shp = gpd.read_file(r'C:\Users\Ion\Downloads\NUPLA\LV95\data/NUPLA_GNNUTZBK.shp')#.iloc[0]['geometry']

poly_list = []
for i in range(len(study_area_shp)):
    poly = study_area_shp.iloc[i]['geometry']
    poly_list.append(poly)
u = cascaded_union(poly_list)

df = pd.DataFrame({'geometry': u})
u = gpd.geodataframe(u)
u.to_file(r'C:\Users\Ion\TFM\data\study_areas/bern/bern_o.shp')

from shapely.geometry import mapping, Polygon, MultiPolygon
import fiona

# Define a polygon feature geometry with one attribute
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'},
}

# Write a new Shapefile
with fiona.open(r'C:\Users\Ion\TFM\data\study_areas/bern/bern_o.shp', 'w', 'ESRI Shapefile', schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    c.write({
        'geometry': mapping(u),
        'properties': {'id': 123},
    })

# luzern stadt
path = r'C:\Users\Ion\TFM\data\study_areas/bern/bern_o.shp'
n_path = []
for letter in path:
    if letter == "\\":
        n_path.append('/')
    else:
        n_path.append(letter)
print(''.join(n_path))


import os

area = 'stgallen'
study_area_dir= r"C:/Users/Ion/TFM/data/study_areas/" + str(area)
file = open(str(study_area_dir) + "/stats_basic.pkl", 'rb')
nodes_betw = pickle.load(file)
areas_list = [dI for dI in os.listdir(r"C:/Users/Ion/TFM/data/study_areas/") if os.path.isdir(os.path.join(r"C:/Users/Ion/TFM/data/study_areas/", dI))]

for area in areas_list:
    study_area_dir = r"C:/Users/Ion/TFM/data/study_areas/" + str(area)
    if os.path.isfile(str(study_area_dir) + "/stats_basic.pkl") == True:
        print(area)

def save():
    save = False
    if save:
        print('asdf')

save()


gdf = gpd.read_file(r"C:/Users/Ion/TFM/data/study_areas/chur/scenarios.shp")
chur = gdf


file = open(r'C:/Users/Ion/TFM/data/network_graphs/ch_nodes_dict2056.pkl', 'rb')
nodes_dict = pickle.load(file)