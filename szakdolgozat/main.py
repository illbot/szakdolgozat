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
import notebooks.missing_link_prediction as mlp

# input beolvasott gráfok, 
def framework():
    # 10 - 90% fairness goodness
    # 10 - 90% éltörlés
    pass

def calculate_errors(deleted_edges, non_existent_edges, treshold, type):
    rmse_result = []
    auc_result = []
    pearson_result = []

    print(len(deleted_edges))
    print(len(non_existent_edges))

    for i in range(len(deleted_edges)):
        if type == "fairness goodness":
            rmse_result.append(lp.calculate_error(deleted_edges[i], resultType="", treshold=treshold))
            pearson_result.append(lp.pearson_correlation(deleted_edges[i], treshold, resultType=""))
        else:
            rmse_result.append(lp.calculate_error(deleted_edges[i], resultType="Similarity", treshold=treshold))
            auc_result.append(lp.calculate_AUC(deleted_edges[i], non_existent_edges[i]))

    if type == "fairness goodness":
        return lp.avg(rmse_result), lp.avg(pearson_result)#, lp.avg(auc_result),
    else:
        return lp.avg(rmse_result), lp.avg(auc_result)


if __name__ == '__main__':
    #G = txtreader.from_csv("BTCAlphaNet.csv")
    #G = txtreader.from_txt("BTCAlphaNet.csv")
    #print(type(G))
    #start = time.perf_counter()


    G = nx.karate_club_graph()


    G2 = G.copy()
    print("cpu count: ", multiprocessing.cpu_count())
    #result = lp.draw_plots(G2, process_number=10)

    #deleted_edges, non_existent_edges = lp.lp_constant_p(G2,p=0.1,alpha=0.1, type=mlp.missing_link_algorithm)

    lp.attribute_approx(G)

    #print(deleted_edges)
    treshold=G.number_of_nodes() * 0.25

    #print(calculate_errors(deleted_edges, non_existent_edges, treshold, type="fairnes"))

    end = time.perf_counter()

    #print("Estimated time "+str(end-start)+", in minutes "+str((end-start)/60))

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    #with open(f'result-{date}.txt', 'w') as f:
    #    f.write(json.dumps(result))


