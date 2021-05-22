import random

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


def attibute_approx_plot(resultdata, xlabel, ylabel, title):
    plt.figure()
    alpha_p_v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    rmse_data = []
    for alpha in alpha_p_v:
        rmse_data.append(resultdata[alpha])

    print(rmse_data)

    plt.plot(alpha_p_v, rmse_data)

    plt.xlabel(xlabel)
    # Set the y axis label of the current axis.
    plt.ylabel(ylabel)
    # Set a title of the current axes.
    plt.title(title)

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
    #LINK PREDIKCIOK
    #--------------------------------FACEBOOK------------------------------------
    """
    G = txtreader.facebook("facebook_combined.txt")
    #rmse = lp.attribute_approx(G)

    rmse = lp.run_paralell_attribute_approx(G, v1_func=nx.closeness_centrality, v2_func=nx.degree_centrality, attr_func=nx.diameter)

    print(rmse)

    attibute_approx_plot(rmse, "alpha values", "RMSE values", "Facebook attribute approx RMSE")

    #rmse, auc, pearson = lp.run_paralell_lp(G, func=mlp.missing_link_algorithm, process_number=6)

    #selected_nodes = random.sample(G.nodes(), k=int(G.number_of_nodes()*0.9))

    #H=G.subgraph(selected_nodes)

    #giant = max(nx.connected_components(H), key=len)

    #rmse, auc, pearson = lp.run_paralell_lp(giant, func=fg.compute_fairness_goodness, process_number=4)

    #make_plot(rmse, "p values", "RMSE values", "Facebook RMSE")
    #make_plot(auc, "p values", "AUC values", "Facebook AUC")

    """

    #-------------------------------BTCAlphaNet--------------------------------
    """
    G = txtreader.alpha("BTCAlphaNet.csv")

    #rmse = lp.attribute_approx(G)
    #rmse = lp.run_paralell_attribute_approx(G)

    rmse, auc, pearson = lp.run_paralell_lp(G, func=fg.compute_fairness_goodness, process_number=4)
    
    #selected_nodes = random.sample(G.nodes(), k=int(G.number_of_nodes()*0.8))
    #H=G.subgraph(selected_nodes)

    #giant = max(nx.strongly_connected_components(H), key=len)

    #G2 = G.subgraph(giant)

    #deleted_edges, non_exisctent = lp.lp_constant_p(G2, p=0.1, alpha=0.1)

    #rmse, auc, pearson = lp.run_paralell_lp(G2, func=fg.compute_fairness_goodness, process_number=4)

    make_plot(rmse, "p values", "RMSE values", "Bitcoin Alpha RMSE")
    make_plot(pearson, "p values", "Pearson values", "Bitcoin Alpha Pearson")
    """

    #-------------------------------BTCOtcNet--------------------------------
    """
    G = txtreader.otc("soc-sign-bitcoinotc.csv")
    print(G.number_of_nodes())
    print(G.number_of_edges())

    rmse, auc, pearson = lp.run_paralell_lp(G, func=fg.compute_fairness_goodness, process_number=1)

    make_plot(rmse, "p values", "RMSE values", "Bitcoin OTC RMSE")
    make_plot(pearson, "p values", "Pearson values", "Bitcoin OTC Pearson")
    """

    #------------------------------KarateKlub----------------------------------
    """
    G= nx.karate_club_graph()
    
    print(G.number_of_nodes())
    print(G.number_of_edges())
    rmse, auc, pearson = lp.run_paralell_lp(G, func=mlp.missing_link_algorithm, process_number=10)

    make_plot(rmse, "p values", "RMSE values", "Karate RMSE")
    make_plot(auc, "p values", "AUC values", "Karate AUC")

    """

    # Attribute Approximatiom
    #-----------------------------Facebook-----------------------------------------
    """
    G = txtreader.facebook("facebook_combined.txt")
    # rmse = lp.attribute_approx(G)

    rmse = lp.run_paralell_attribute_approx(G, v1_func=nx.closeness_centrality, v2_func=nx.degree_centrality,
                                            attr_func=nx.diameter)

    print(rmse)

    attibute_approx_plot(rmse, "alpha values", "RMSE values", "Facebook attribute approx RMSE")
    
    """
    #--------------------Barabasi-Albert--------------------------
    #---------300--------

    G = nx.barabasi_albert_graph(300, 5)

    rmse = lp.run_paralell_attribute_approx(G)

    attibute_approx_plot(rmse, "alpha values", "RMSE values", "Barabasi-Albert 300 node attribute approx RMSE")

    #rmse = lp.run_paralell_attribute_approx(G, attr_func=nx.attribute_assortativity_coefficient)
    #attibute_approx_plot(rmse, "alpha values", "RMSE values", "Facebook attribute approx RMSE")

    #rmse =lp.attribute_approx(G, attributum_method=nx.triangles)
    #rmse = lp.run_paralell_attribute_approx(G, attr_func=nx.triangles)
    #print(rmse)

    #attibute_approx_plot(rmse, "alpha values", "RMSE values", "Facebook attribute approx RMSE")

    # ---------400--------

    G = nx.barabasi_albert_graph(400, 5)

    rmse = lp.run_paralell_attribute_approx(G)

    attibute_approx_plot(rmse, "alpha values", "RMSE values", "Barabasi-Albert 400 node attribute approx RMSE")
    # ---------500--------

    G = nx.barabasi_albert_graph(500, 5)

    rmse = lp.run_paralell_attribute_approx(G)

    attibute_approx_plot(rmse, "alpha values", "RMSE values", "Barabasi-Albert 500 node attribute approx RMSE")

    # --------------------Erdos-Renyi--------------------------

    # ---------300--------
    G = nx.erdos_renyi_graph(300, 0.5)

    rmse = lp.run_paralell_attribute_approx(G)
    attibute_approx_plot(rmse, "alpha values", "RMSE values", "Erdos-Renyi 300 node attribute approx RMSE")

    #rmse = lp.run_paralell_attribute_approx(G, attr_func=nx.attribute_assortativity_coefficient)
    #attibute_approx_plot(rmse, "alpha values", "RMSE values", "Facebook attribute approx RMSE")

    #rmse = lp.run_paralell_attribute_approx(G, attr_func=nx.triangles)
    #attibute_approx_plot(rmse, "alpha values", "RMSE values", "Facebook attribute approx RMSE")

    # ---------400--------

    G = nx.erdos_renyi_graph(400, 0.5)

    rmse = lp.run_paralell_attribute_approx(G)

    attibute_approx_plot(rmse, "alpha values", "RMSE values", "Erdos-Renyi 400 node attribute approx RMSE")

    # ---------500--------
    G = nx.erdos_renyi_graph(500, 0.5)

    rmse = lp.run_paralell_attribute_approx(G)

    attibute_approx_plot(rmse, "alpha values", "RMSE values", "Erdos-Renyi 500 node attribute approx RMSE")




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




