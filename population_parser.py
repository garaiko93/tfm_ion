import gzip
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
import os
import re
import datetime
from attrdf_manip import take_avg

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

def export_dict(dict, file_name, out_path, file_number):
    # EXPORT dicts TO FILEs
    with open(str(out_path) + '/' + str(file_name) + '_' + str(file_number) + '.pkl', 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)
    df = pd.DataFrame.from_dict(dict, orient='index')
    # df.to_csv(str(out_path) + '/' + str(file_name) + '_' + str(file_number) + '.csv', sep=",", index=True,
    #           index_label=['person_id', 'trip', 'leg'])

    if len(df.index[0]) == 3:
        df.to_csv(str(out_path) + '/' + str(file_name) + '_' + str(file_number) + '.csv', sep=",", index=True, index_label=['person_id','trip','leg'])
    elif len(df.index[0]) == 2:
        df.to_csv(str(out_path) + '/' + str(file_name) + '_' + str(file_number) + '.csv', sep=",", index=True, index_label=['person_id','trip'])
    else:
        df.to_csv(str(out_path) + '/' + str(file_name) + '_' + str(file_number) + '.csv', sep=",", index=True, index_label=['person_id'])


# PARSE NETWORK XML FILE BY A PARSER
def population_parser_etree(pop_folder,out_folder,scenario):
    # define path and file variables
    print(datetime.datetime.now(), scenario)
    out_path = str(out_folder) + '/' + str(scenario)
    pop_file = str(pop_folder) + '/' + str(scenario) + '/switzerland_population.xml.gz'
    # check if out directory exists and create it if not
    if not os.path.exists(str(out_path)):
        os.makedirs(str(out_path))
        print(datetime.datetime.now(), 'Directory created')
    else:
        print(datetime.datetime.now(), 'Directory exists')

    # open population file and parse
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
                print(datetime.datetime.now(), person_id)
            for child in elem:
                activity_leg.append(child)
        if divmod(len(activity_leg),2)[1] != 1:
            print(datetime.datetime.now(), person_id)
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

    print(datetime.datetime.now(), str('Population_attributes contains attributes of ')+str(len(population_attributes)) + str(' persons.'))
    print(datetime.datetime.now(), str('Population_plans contains information of ')+str(len(population_plans)) + str(' trips.'))

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
        print(datetime.datetime.now(), loc, len(loc_df))

# data:
# -there is only one 'plan' element per person
# -plans are described as: activity(origin)-route-activity(destination)
# -plan: selected='no' (not found)
# -why attributes length = 7984 and unique id population_plans is 7533? this is people that stays at home

# -----------------------------------------------------------------------------
# MAIN FUNCTION OF SCRIPT: checks if a group of scenarios or a specific one is passed to then call 'check_pop_file'
# -----------------------------------------------------------------------------
def population_parser_setup(scenario_area_dir, scenario_area=None):
    print(datetime.datetime.now(), 'Population parser setup begins ...')
    if scenario_area:
        check_pop_file(scenario_area_dir, scenario_area)
    else:
        scr_list = [dI for dI in os.listdir(scenario_area_dir) if os.path.isdir(os.path.join(scenario_area_dir, dI))]
        for scr in scr_list:
            check_pop_file(scenario_area_dir, scr)

# -----------------------------------------------------------------------------
# SECOND function of script: assigns out_path and finds population.xml file
# -----------------------------------------------------------------------------
def check_pop_file(scenario_area_dir, scenario_area):
    out_path = str(scenario_area_dir) + '/' + str(scenario_area) + '/population_db'
    swiss_scr = os.path.isfile(
        str(scenario_area_dir) + '/' + str(scenario_area) + '/switzerland_population.xml.gz')
    study_area_scr = os.path.isfile(
        str(scenario_area_dir) + '/' + str(scenario_area) + '/' + str(scenario_area) + '_population.xml.gz')

    if swiss_scr:
        pop_file = str(scenario_area_dir) + '/' + str(scenario_area) + '/switzerland_population.xml.gz'
        print(datetime.datetime.now(), 'Population file found in scenario folder as: switzerland_population.xml.gz')
        population_parser_line(out_path, pop_file, scenario_area, None)
    elif study_area_scr:
        pop_file = str(scenario_area_dir) + '/' + str(scenario_area) + '/' + str(
            scenario_area) + '_population.xml.gz'
        print(datetime.datetime.now(), 'Population file found in scenario folder as: ' + str(scenario_area) + '_population.xml.gz')
        population_parser_line(out_path, pop_file, scenario_area, scenario_area_dir)
    else:
        raise Exception("Scenario folder has no population.xml file!")

# -----------------------------------------------------------------------------
# THIRD function: it parses the population file found before
# -----------------------------------------------------------------------------
def population_parser_line(out_path, pop_file, scenario_area=None, attr_table_dir=None):
    # check if out_path exists and create it if not
    if not os.path.exists(str(out_path)):
        os.makedirs(str(out_path))
        print(datetime.datetime.now(), 'Directory created')
    else:
        print(datetime.datetime.now(), 'Directory exists')

    population_attributes = {}
    population_plans = {}
    facility_dict = {}
    attribute_check = 0
    plan_check = 0
    activity_check = 0
    person_count = 0
    file_number = 0
    attr_count = 0
    plans_count = 0

    if os.path.isfile(str(out_path)+"/loc_freight_loading.csv") == False:
        print(datetime.datetime.now(), 'Parsing population ...')
        with gzip.open(pop_file) as f:
            #     reading line by line the 'nodes' file created at the beginning, data for each node fulfilling the conditions are stored for the output
            for line in f:
                if b"<person" in line:
                    m = re.search(rb'id="(.+)"', line)
                    if m:
                        person_id = m.group(1).decode('utf-8')
                        attribute_check = 1
                        person_attr = {}
                        # print(person_id)
                # extract attributes from person
                if attribute_check == 1:
                    if b"<attribute " in line:
                        m = re.search(rb'name="(.+)" class="(.+)" >(.+)</attribute', line)
                        if m:
                            name = m.group(1).decode('utf-8')
                            value = m.group(3).decode('utf-8')
                            person_attr[name] = value
                    if b"</attributes>" in line:
                        population_attributes[person_id] = person_attr
                        attribute_check = 0
                # extract plans from person
                if b"<plan " in line:
                    act_leg_list = []
                    m = re.search(rb'selected="(.+)"', line)
                    if m:
                        selected = m.group(1).decode('utf-8')
                        plan_check = 1
                if plan_check == 1:
                    if b"<activity " in line:
                        # m = re.search(rb'type="(.+)" link="(.+)" facility="(.+)" x="([+-]?\d+(?:\.\d+)?)" y="([+-]?\d+(?:\.\d+)?)"', line)
                        # m = re.search(rb'type="(.+)" link="(.+)" x="([+-]?\d+(?:\.\d+)?)" y="([+-]?\d+(?:\.\d+)?)"', line)

                        type = re.search(rb'type="([^"]*)"', line).group(1).decode('utf-8')
                        link = re.search(rb'link="([^"]*)"', line).group(1).decode('utf-8')
                        x = re.search(rb'x="([^"]*)"', line).group(1).decode('utf-8')
                        y = re.search(rb' y="([^"]*)"', line).group(1).decode('utf-8')
                        activity_check = 1

                        # facility regex match
                        # m = re.search(rb'facility="([A-Za-z]+[\d@]+[\w@]*|[\d@]+[A-Za-z]+[\w@]*|[+-]?\d+(?:\.\d+)?)"', line)
                        m = re.search(rb'facility="([^"]*)"', line)
                        if m:
                            facility = m.group(1).decode('utf-8')
                        else:
                            facility = str(type) + str(link)

                        # add record to facility dictionary
                        if facility not in list(facility_dict):
                            persons = [person_id]
                            facility_dict[facility] = (type, x, y, link, persons, len(persons))
                        else:
                            persons = facility_dict[facility][4]
                            if person_id not in persons:
                                persons.append(person_id)
                                facility_dict[facility] = (type, x, y, link, persons, len(persons))

                    if (b"<attribute " in line) == True and activity_check == 1:
                        m = re.search(rb'name="(.+)" class="(.+)" >(.+)</attribute', line)
                        if m:
                            # name = m.group(1).decode('utf-8')
                            value = m.group(3).decode('utf-8')
                    if (b"</activity>" in line) == True and activity_check == 1:
                        act_leg_list.append([type, facility, x, y, value, 'activity'])
                        # act_leg_list.append([type, 0, x, y, value, 'activity'])
                        activity_check = 0
                    if b"<leg " in line:
                        m = re.search(rb'mode="(.+)" dep_time="(.+)" trav_time="(.+)"', line)
                        if m:
                            mode = m.group(1).decode('utf-8')
                            dep_time = m.group(2).decode('utf-8')
                            # trav_time = m.group(3).decode('utf-8')
                    if b"<route " in line:
                        # m = re.search(rb'type="(.+)" start_link="(.+)" end_link="(.+)" trav_time="(.+)" '
                        #               rb'distance="([+-]?\d+(?:\.\d+)?)" >([+-]?\d+(?:\.\d+)?)</route', line)
                        m = re.search(rb'type="(.+)" start_link="(.+)" end_link="(.+)" trav_time="(.+)" distance="([+-]?\d+(?:\.\d+)?)"', line)
                        if m:
                            type = m.group(1).decode('utf-8')
                            start_link = m.group(2).decode('utf-8')
                            end_link = m.group(3).decode('utf-8')
                            trav_time = m.group(4).decode('utf-8')
                            distance = m.group(5).decode('utf-8')
                            # vehiclerefid = m.group(6).decode('utf-8')
                            # route = m.group(6).decode('utf-8')
                            act_leg_list.append([mode, dep_time, trav_time, type, start_link, end_link, distance, 'leg'])
                    if b"</plan>" in line:
                        plan_check = 0
                        act_1_dict = {}
                        act_2_dict = {}
                        leg_dict = {}
                        home_dict = {}
                        if len(act_leg_list) < 3:
                            act = act_leg_list[0]
                            home_dict['from_type'] = act[0]
                            home_dict['from_facility'] = act[1]
                            # home_dict['from_x'] = act[2]
                            # home_dict['from_y'] = act[3]
                            # home_dict['ov_guteklasse'] = act[4]
                            home_dict['stays_home'] = 'true'
                            home_dict['n_trips'] = 0
                            population_plans[person_id, 0, 0] = home_dict
                        else:
                            for i in range(0,len(act_leg_list)-1,2):
                                if act_leg_list[0][-1] == 'leg':
                                    continue
                                n_trip = int(i / 2)
                                n_leg = 0

                                act_1 = act_leg_list[i]
                                act_1_dict['from_type'] = act_1[0]
                                act_1_dict['from_facility'] = act_1[1]
                                # act_1_dict['from_x'] = act_1[2]
                                # act_1_dict['from_y'] = act_1[3]
                                # act_1_dict['ov_guteklasse'] = act_1[4]

                                leg = act_leg_list[i+1]
                                leg_dict['mode'] = leg[0]
                                leg_dict['dep_time'] = leg[1]
                                leg_dict['trav_time'] = leg[2]
                                leg_dict['type'] = leg[3]
                                # leg_dict['start_link'] = leg[4]
                                # leg_dict['end_link'] = leg[5]
                                leg_dict['distance'] = leg[6]
                                # leg_dict['vehicleRefId'] = leg[7]
                                # leg_dict['route'] = leg[8]

                                act_2 = act_leg_list[i+2]
                                if act_2[-1] == 'leg':
                                    plans_dict = {**act_1_dict, **leg_dict}
                                    population_plans[person_id, n_trip, n_leg] = plans_dict
                                    n_leg += 1

                                if n_leg != 0:
                                    leg_2 = act_leg_list[i+2]
                                    leg_dict = {}
                                    leg_dict['mode'] = leg_2[0]
                                    leg_dict['dep_time'] = leg_2[1]
                                    leg_dict['trav_time'] = leg_2[2]
                                    leg_dict['type'] = leg_2[3]
                                    # leg_dict['start_link'] = leg_2[4]
                                    # leg_dict['end_link'] = leg_2[5]
                                    leg_dict['distance'] = leg_2[6]

                                    act_2 = act_leg_list[i+3]
                                    act_2_dict['to_type'] = act_2[0]
                                    act_2_dict['to_facility'] = act_2[1]
                                    # act_2_dict['to_x'] = act_2[2]
                                    # act_2_dict['to_y'] = act_2[3]

                                    plans_dict = {**leg_dict, **act_2_dict}
                                    plans_dict['stays_home'] = 'false'
                                    plans_dict['n_trips'] = divmod(len(act_leg_list), 2)[0]
                                    population_plans[person_id, n_trip, n_leg] = plans_dict

                                act_2_dict['to_type'] = act_2[0]
                                act_2_dict['to_facility'] = act_2[1]
                                # act_2_dict['to_x'] = act_2[2]
                                # act_2_dict['to_y'] = act_2[3]

                                plans_dict = {**act_1_dict, **leg_dict, **act_2_dict}
                                plans_dict['stays_home'] = 'false'
                                plans_dict['n_trips'] = divmod(len(act_leg_list), 2)[0]
                                population_plans[person_id, n_trip, n_leg] = plans_dict
                        # this part of the code splits the output in different files, to avoid memory problems
                        person_count +=1
                if (b"</population>" in line) == True or person_count == 85000:
                    # EXPORT DICT on .pkl and .csv
                    export_dict(population_attributes, 'population_attributes', out_path, file_number)
                    export_dict(population_plans, 'population_plans', out_path, file_number)
                    with open(str(out_path) + '/facilities_dict.pkl', 'wb') as f2:
                        pickle.dump(facility_dict, f2, pickle.HIGHEST_PROTOCOL)
                    attr_count += len(population_attributes)
                    plans_count += len(population_plans)
                    person_count = 0
                    file_number += 1
                    population_attributes = {}
                    population_plans = {}

        if attr_table_dir:
            attr_df = pd.read_csv(str(attr_table_dir) + '/attribute_table.csv', sep=",", index_col='attributes',
                                  dtype=object)
            print(datetime.datetime.now(), 'Attributes table loaded successfully.')
            attr_df.at['population', scenario_area] = attr_count
            attr_df.at['trips', scenario_area] = plans_count
            take_avg(attr_df)
            # attr_df.to_csv(str(attr_table_dir) + "/attribute_table.csv", sep=",", index=True, index_label=['attributes'])
        print(datetime.datetime.now(), str('Population_attributes contains attributes of ') + str(attr_count) + str(' persons.'))
        print(datetime.datetime.now(), str('Population_plans contains information of ') + str(plans_count) + str(' trips.'))

        # create df with location of all facilities with groupby.list person_id
        for i in range(0, file_number):
            print(datetime.datetime.now(), 'file ' + str(i))
            file = open(str(out_path) + '/' + 'population_plans_' + str(i) + '.pkl', 'rb')
            population_plans = pickle.load(file)
            df = pd.DataFrame.from_dict(population_plans, orient='index')
            faci_df = pd.DataFrame.from_dict(facility_dict, orient='index')
            faci_df = faci_df.reset_index()
            faci_df.columns = ['facility_id', 'type', 'x', 'y', 'link', 'person_ids', 'n_persons']
            for loc in df.from_type.unique():
                loc_file = str(out_path) + '/loc_' + str(loc) + '.csv'
                loc_df = faci_df[faci_df['type'] == loc][['facility_id', 'x', 'y', 'link', 'person_ids', 'n_persons']]
                loc_df.to_csv(loc_file, sep=",", index=False)
                print(datetime.datetime.now(), loc, len(loc_df))



        # home_loc = pd.read_csv(str(out_path) + "/loc_home.csv")
        # home_loc = area_fac
        # area_fac.head()
        # area_fac.columns
        # for index, row in home_loc.iterrows():
        #
        # print(home)

# population_parser_line(r'C:/Users/Ion/TFM/data/study_areas', 'switzerland_1pm', None)
    # study_area_dir = r'C:/Users/Ion/TFM/data/study_areas'
    # pop_folder = r'C:\Users\Ion\TFM\data\scenarios'
    # out_folder = r'C:\Users\Ion\TFM\data\population_db/test'
    # scenario = 'switzerland_1pm'
    # area = None