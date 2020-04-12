import os
import argparse
import datetime
import time

from wt_extract import wt_extract_fc
from wt_plot import plot_fc

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
parser.add_argument('--tfm-dir', dest="tfm_dir", help='path to simulation_output folder')
parser.add_argument('--area', dest="area", help='path to simulation_output folder')
parser.add_argument('--init-fleetsize', dest="init_fleet_size", help='path to simulation_output folder')
parser.add_argument('--fleet-incrpersim', dest="fleet_incr_per_sim", help='path to simulation_output folder')
parser.add_argument('--init-fleetsizeincr', dest="init_fleet_size_incr", help='path to simulation_output folder')
parser.add_argument('--wt-min', dest="wt_min", help='path to simulation_output folder')
args = parser.parse_args()

os.system('conda activate pycharm_basic')

tfm_dir = args.tfm_dir
area = args.area
init_fleet_size = int(args.init_fleet_size)
fleet_incr_per_sim = int(args.fleet_incr_per_sim)
init_fleet_size_incr = int(args.init_fleet_size_incr)
wt_min = int(args.wt_min)

print(datetime.datetime.now(), 'Start simulation for ' + str(area) + ' area.')

for av_share in [0.2, 0.4, 0.6, 0.8, 1.0]:
    print('----------------------------------------------------------------------')
    print(datetime.datetime.now(), 'av_share: ' + str(av_share))
    print('----------------------------------------------------------------------')
    print(av_share, type(av_share))
    wt_avg = 1000
    fleet_incr = fleet_incr_per_sim
    fleet_size = init_fleet_size
    AREA_PATH = str(tfm_dir) + '/data/study_areas/' + str(area)

    while float(wt_avg) > float(wt_min):
        FOLDER = str(AREA_PATH) + '/simulations/' + str(av_share) + '/' + str(fleet_size)
        if not os.path.exists(str(FOLDER)):
            os.makedirs(str(FOLDER))
            print(datetime.datetime.now(), 'Directory created')
        else:
            print(datetime.datetime.now(), 'Directory exists')

        # os.system('cd ' + str(FOLDER))
        os.chdir(FOLDER)
        OUT_FILE = str(FOLDER) + '/simulation_output/av_passenger_rides.csv'
        if not os.path.isfile(OUT_FILE):
            print(datetime.datetime.now(), 'Simulating area: ' + str(area) + ' av share: ' + str(av_share) + ' fleet size: ' + str(fleet_size))
            start_time = time.time()
            java_command = 'java -cp ' + str(
                tfm_dir) + '/simulations/eqasim-java/classes/artifacts/eqasim_jar/eqasim.jar org.eqasim.examples.zurich_av.RunSimulation --config-path ' + str(
                AREA_PATH) + '/simulations/' + str(area) + '_config.xml --fleet-size ' + str(
                fleet_size) + ' --av-share ' + str(av_share) + ' 1> sim_output.txt 2>&1'
            # print(java_command)
            os.system(java_command)
            print(datetime.datetime.now(), "simulation time was:  %s seconds " % (time.time() - start_time))

        wt_avg = wt_extract_fc(AREA_PATH, str(fleet_size), str(av_share))
        print(datetime.datetime.now(), 'waiting time average: ' + str(wt_avg))

        # Update fleet size for new simulation
        fleet_size = fleet_size + fleet_incr
        print('----------------------------------------------------------------------')

    # one improve could be to change the fleet size increase for the low avshare values (0.2 and 0.4) which needs to be closer
    # and increase the one of the av share high values of 0.8 and 1.0
    # Update initial fleet size for new av-share
    init_fleet_size = init_fleet_size + init_fleet_size_incr

# Finally, plot results of all simulatinos for the different av-share values
plot_fc(AREA_PATH)

print('----------------------------------------------------------------------')
print(datetime.datetime.now(), 'Simulation process finished, output files and plot in simulations folder')
print('----------------------------------------------------------------------')

