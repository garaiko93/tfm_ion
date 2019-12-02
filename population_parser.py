import gzip
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
import os

def extract_activity(activity,from_to):
    act_list = activity.attrib
    act_dict = {}
    for at in list(act_list):
        if at not in ['link', 'start_time', 'end_time']:
            act_dict[str(from_to) + '_' + str(at)] = act_list[at]
    try:
        ov_guteklasse = activity[0][0]
        act_dict['ov_guteklasse'] = ov_guteklasse.text
    except:
        act_dict['ov_guteklasse'] = ''
    return act_dict

def export_dict(dict,file_name,out_path):
    # EXPORT dicts TO FILEs
    with open(str(out_path) + '/'+str(file_name) + '.pkl', 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)
    df = pd.DataFrame.from_dict(dict, orient='index')
    if len(df.index[0]) == 2:
        df.to_csv(str(out_path) + '/' + str(file_name) + '.csv', sep=",", index=True, index_label=['person_id','trip'])
    else:
        df.to_csv(str(out_path) + '/'+str(file_name)+'.csv', sep=",", index=True, index_label=['person_id'])

# pop_file = r'C:\Users\Ion\TFM\data\scenarios\switzerland_1pm\switzerland_population.xml.gz'
# out_path = r'C:\Users\Ion\TFM\data\population_db'
# PARSE NETWORK XML FILE BY A PARSER
def population_parser(pop_file,out_path):
    if not os.path.exists(str(out_path)):
        os.makedirs(str(out_path))
        print('Directory created')
    else:
        print('Directory exists')

    with gzip.open(pop_file) as f:
        tree = ET.parse(f)
        root = tree.getroot()

    # create different databases in a dictionary shape (key,value):
    population_attributes = {}
    population_plans = {}
    count_p = 0
    # element iteration through population file
    for person in root.findall('person'):
        person_id = person.get('id')

        # extraction of attributes of person
        attrib_dict = {}
        for elem in person.findall('attributes'):
            for attribute in elem:
                attrib_dict[attribute.get('name')] = attribute.text
            population_attributes[person_id] = attrib_dict

        # extraction of plan activities
        activity_leg = []
        for elem in person.findall('plan'):
            selected = elem.get('selected')
            if selected != 'yes':
                print(person_id)
            for child in elem:
                activity_leg.append(child)
        if divmod(len(activity_leg),2)[1] != 1:
            print(person_id)
        if len(activity_leg) < 3:
            home_dict = extract_activity(activity_leg[0],'from')
            home_dict['stays_home'] = 'true'
            home_dict['n_trips'] = 0
            population_plans[person_id,0] = home_dict
        for i in range(0,len(activity_leg)-1,2):
            n_trip = int(i/2)
            act_1_dict = extract_activity(activity_leg[i],'from')

            leg = activity_leg[i+1]
            leg_dict = leg.attrib
            try:
                route = leg[0]
                route_dict = route.attrib
                route_dict['route'] = route.text
                leg_dict = {**leg_dict, **route_dict}
            except:
                pass

            act_2_dict = extract_activity(activity_leg[i+2],'to')

            plans_dict = {**act_1_dict, **leg_dict, **act_2_dict}
            plans_dict['stays_home'] = 'false'
            plans_dict['n_trips'] = divmod(len(activity_leg),2)[0]
            population_plans[person_id,n_trip] = plans_dict

    # EXPORT DICT on .pkl and .csv
    export_dict(population_attributes,'population_attributes',out_path)
    export_dict(population_plans, 'population_plans',out_path)

    print(str('Population_attributes contains attributes of ')+str(len(population_attributes)) + str(' persons.'))
    print(str('Population_plans contains information of ')+str(len(population_plans)) + str(' trips.'))

    # create df with location of all facilities with groupby.list person_id
    df = pd.DataFrame.from_dict(population_plans, orient='index')
    for loc in df.from_type.unique():
        loc_df = df[df['from_type'] == loc][['from_x', 'from_y']]
        loc_df.index.names = ['person_id', 'trip']
        loc_df = loc_df.reset_index()
        loc_df = loc_df[['person_id', 'from_x', 'from_y']].drop_duplicates()
        loc_df = loc_df.groupby(['from_x', 'from_y'])['person_id'].apply(list)
        loc_df.index.names = ['x', 'y']
        loc_df = loc_df.reset_index()
        for i in loc_df.index:
            loc_df.at[i, 'n_persons'] = (len(loc_df.at[i, 'person_id']))
        loc_df.to_csv(str(out_path) + '\loc_' + str(loc) + '.csv', sep=",", index=False)
        print(loc, len(loc_df))

# data:
# -there is only one 'plan' element per person
# -plans are described as: activity(origin)-route-activity(destination)
# -plan: selected='no' (not found)
# -why attributes length = 7984 and unique id population_plans is 7533? this is people that stays at home
