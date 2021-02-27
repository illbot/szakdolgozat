import networkx as nx
import matplotlib.pyplot as plt
import io, os, sys, types
import time

cwd = os.getcwd()
sys.path.insert(0, cwd+'\\notebooks')

import notebooks.create_from_txt as txtreader
import notebooks.link_prediction as lp

if __name__ == '__main__':
    G = txtreader.from_csv("BTCAlphaNet.csv")
    start = time.perf_counter()
    G2 = G.copy()
    result = lp.draw_plots(G2)

    end = time.perf_counter()

    print("Estimated time "+str(end-start)+", in minutes "+str((end-start)/60))

