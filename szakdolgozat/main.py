import networkx as nx
import matplotlib.pyplot as plt
import io, os, sys, types
import time
import datetime
import multiprocessing
import json

cwd = os.getcwd()
sys.path.insert(0, cwd+'\\notebooks')

import notebooks.create_from_txt as txtreader
import notebooks.link_prediction as lp

# input beolvasott gráfok, 
def framework():
    # 10 - 90% fairness goodness
    # 10 - 90% éltörlés
    pass


if __name__ == '__main__':
    G = txtreader.from_csv("BTCAlphaNet.csv")
    start = time.perf_counter()
    G2 = G.copy()
    print("cpu count: ", multiprocessing.cpu_count())
    result = lp.draw_plots(G2, process_number=10)

    end = time.perf_counter()

    print("Estimated time "+str(end-start)+", in minutes "+str((end-start)/60))

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    with open(f'result-{date}.txt', 'w') as f:
        f.write(json.dumps(result))


