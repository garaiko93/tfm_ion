import re
import gzip
import statistics
import time

plans_file = r'C:/Users/Ion/TFM/simulations/auckland_example2/simulation_output/ITERS/it.40/40.plans.xml.gz'
wt_list = []
with gzip.open(plans_file) as f:
    for line in f:
        if b'type="av"' in line:
            m = re.search(rb'"waitingTime":(.+),', line)
            wt = float(m.group(1))
            wt_list.append(wt)
print('Plans in AVs with waitingTime values: ' + str(len(wt_list)))
print('Average of all waitingTime values: ' + str(time.strftime('%H:%M:%S', time.gmtime(statistics.mean(wt_list)))))
