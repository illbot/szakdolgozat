import networkx as nx
import matplotlib.pyplot as plt
import io, os, sys, types
import time
import datetime
import multiprocessing
import json
import matplotlib.pyplot as plt


cwd = os.getcwd()
sys.path.insert(0, cwd+'\\notebooks')

import notebooks.create_from_txt as txtreader
import notebooks.link_prediction as lp
import notebooks.missing_link_prediction as mlp
import fairness_goodness_computation as fg



def make_plot(resultdata, xlabel, ylabel, title):
    plt.figure()
    alpha_p_v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    rmse_data = {
        0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: []
    }
    for alpha in alpha_p_v:
        for p in alpha_p_v:
            rmse_data[alpha].append(resultdata[alpha][p])

    print(rmse_data)

    for alpha in alpha_p_v:
        plt.plot(alpha_p_v, rmse_data[alpha], label= str(alpha))

    plt.xlabel(xlabel)
    # Set the y axis label of the current axis.
    plt.ylabel(ylabel)
    # Set a title of the current axes.
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', title='alpha values')
    plt.show()


def calculate_errors(deleted_edges, non_existent_edges, treshold, type:str):
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
    """
    #--------------------------------FACEBOOK------------------------------------
    G = txtreader.facebook("facebook_combined.txt")
    rmse, auc, pearson = lp.run_paralell_lp(G, func=mlp.missing_link_algorithm, process_number=6)
    
    make_plot(rmse, "p values", "RMSE values", "Facebook RMSE")
    make_plot(auc, "p values", "AUC values", "Facebook AUC")
    """

    """
    #-------------------------------BTCAlphaNet--------------------------------
    G = txtreader.from_csv("BTCAlphaNet.csv")

    rmse, auc, pearson = lp.run_paralell_lp(G, func=lp.lp_constant_p, process_number=6)
    
    make_plot(rmse, "p values", "RMSE values", "Bitcoin Alpha RMSE")
    make_plot(pearson, "p values", "Pearson values", "Bitcoin Alpha Pearson")
    """

    #-------------------------------BTCOtcNet--------------------------------
    G = txtreader.otc("soc-sign-bitcoinotc.csv")
    print(G.number_of_nodes())
    print(G.number_of_edges())
    rmse, auc, pearson = lp.run_paralell_lp(G, func=lp.lp_constant_p, process_number=2)

    make_plot(rmse, "p values", "RMSE values", "Bitcoin OTC RMSE")
    make_plot(pearson, "p values", "Pearson values", "Bitcoin OTC Pearson")

    #G = txtreader.from_csv("BTCAlphaNet.csv")

    #G = txtreader.from_txt("BTCAlphaNet.csv")
    #print(type(G))
    #start = time.perf_counter()

    #print(G.number_of_nodes())
    #print(G.number_of_edges())

    #G = nx.karate_club_graph()

    #lp.plotting("result-2021-05-05-15-38-45.txt")


    #result = lp.draw_plots(G2, process_number=10)

    #deleted_edges, non_existent_edges = lp.lp_constant_p(G2, p=0.6, alpha=0.1, type=mlp.missing_link_algorithm)

    #rmse, auc = calculate_errors(deleted_edges=deleted_edges, non_existent_edges=non_existent_edges, treshold=G.number_of_nodes() * 0.25, type="")

    #print(rmse, auc)

    # FACEBOOK GRAPH


    # matplotlib x = p értékek
    # y1-9 = rmse vagy auc alfánként 1-1 vonal, minden alfához van 0.1-0.9-ig érték


    """lp.attribute_approx(G)

    #print(deleted_edges)
    treshold=G.number_of_nodes() * 0.25

    #print(calculate_errors(deleted_edges, non_existent_edges, treshold, type="fairnes"))

    end = time.perf_counter()

    #print("Estimated time "+str(end-start)+", in minutes "+str((end-start)/60))

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    #with open(f'result-{date}.txt', 'w') as f:
    #    f.write(json.dumps(result))"""


