import argparse
def funct(study_areas,network_graph):
    c = study_areas+network_graph
    print(c)

parser = argparse.ArgumentParser(description='Cut and analyse a graph for a certain input area.')
parser.add_argument('--study-areas', dest="study_areas", type=int, help='path to study areas')
parser.add_argument('--network-graphs', dest="network_graphs", type=int, help="path to network_graphs")

args = parser.parse_args()
print(args)
funct(args.study_areas, args.network_graphs)