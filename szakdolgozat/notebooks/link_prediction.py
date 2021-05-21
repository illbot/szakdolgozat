import networkx as nx
import numpy as np
import math
import random
import fairness_goodness_computation as fg
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import threading
import multiprocessing
import concurrent.futures
import queue
import time
from multiprocessing.managers import SharedMemoryManager
import json
import missing_link_prediction as mlp
import main


def calculate_AUC(deleted_edges, non_existent_edges):
    N = 25
    deleted_list = []
    non_existent_list = []
    for i in range(N):
        deleted_list.append(random.choice(deleted_edges))
        non_existent_list.append(random.choice(non_existent_edges))

    n1 = 0
    n2 = 0
    for i in range(len(deleted_list)):
        if deleted_list[i][3] > non_existent_list[i][3]:
            n1 = n1+1
        elif deleted_list[i][3] == non_existent_list[i][3]:
            n2 = n2 + 1

    return (n1 + 0.5*n2)/N


def pearson_correlation(deleted_edges, treshold, resultType):
    if resultType == "Similarity":
        list1 = []
        for edge in deleted_edges:
            edge1 = list(edge)
            if edge1[3] > treshold:
                edge1[3] = 1
                list1.append(tuple(edge1))
            else:
                edge1[3] = 0
                list1.append(tuple(edge1))
        deleted_edges = list1

    element_3 = []
    element_4 = []
    for edge in deleted_edges:
        element_3.append(edge[2])
        element_4.append(edge[3])

    print("element3: ", np.array(element_3))
    print("element4: ", np.array(element_4))
    print("pearson:", np.corrcoef(np.array(element_3), np.array(element_4)))
    return np.corrcoef(np.array(element_3), np.array(element_4))[0][1]


def calculate_error(deleted_edges, resultType, treshold, Type="RMSE"):
    if Type == "RMSE":
        if resultType == "Similarity":
            list1 = []
            for edge in deleted_edges:
                edge1 = list(edge)
                if edge1[3] > treshold:
                    edge1[3] = 1
                    list1.append(tuple(edge1))
                else:
                    edge1[3] = 0
                    list1.append(tuple(edge1))
            deleted_edges = list1

        old_w = []
        predicted_w = []

        for edge in deleted_edges:
            if not math.isnan(edge[3]):
                old_w.append(edge[2])
                predicted_w.append(edge[3])

        return RMSE(old_w, predicted_w)
    else:
        pass
    # for edge in deleted_edges:
    #    edge[2] edge[3]-bol 2 vektor


def RMSE(actual, predicted):
    result = None
    predicted_real = []
    for weight in predicted:
        if isinstance(weight, complex):
            predicted_real.append(weight.real)

    if len(predicted_real) > 0:
        result = math.sqrt(mean_squared_error(actual, predicted_real))
    else:
        result = math.sqrt(mean_squared_error(actual, predicted))
    return result


def avg(list1):
    return (sum(list1) / len(list1))


def lp(G, alpha=None):
    """G: graph
       p: percentage of the edges to be deleted in every iteration exp.: 0.1 = 10%
       iteration: how many times you want to delete x percentage of the edges
    """
    # WIP
    # A G referencia lehet h érdemes elöbb másolni a gráfot h az eredetiből ne töröljünk éleket
    if alpha is not None:
        if alpha < 0 and alpha > 1:
            raise Exception("alpha parameter needs to be between 0 and 1")

    XPERCENT = math.floor(G.size() * p)  # p percent of the edges
    Available_edges = list(G.edges.data('weight', default=0))  # get all edges with weights
    Deleted_edges = []  # deleted edges

    # p_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # 10 és 90% között végig megyünk

    # for p in p_values:
    #    rmse = []
    # 1...25 for ciklus
    # G2 = G.copy()
    #        g.remove_edges_from(random.sample(g.edges(),k=int(p*g.number_of_edges())))
    #         fairness, goodness = fg.compute_fairness_goodness(G) # 0 elem fairness * 1 elem goodness

    #         if alpha is None: # If this method called without alpha param.
    #         for edge in Deleted_edges: # all deleted edges for fairness goodness
    #             _from = edge[0]
    #             _to =edge[1]
    #             edge = (*edge, (fairness[_from]*goodness[_to])) # (from, to, old_weight, new_weight)
    #             print(edge)
    #         temp_rmse.append(calculate_error (Deleted_edges, "RMSE"))
    #   rmse.append(atlag(temp_rmse)) ebből a kép x tengely p, y tengely rmse

    p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rmse = []

    for p in p_values:
        temp_rmse = []
        for i in range(25):
            G2 = G.copy()
            updated_deleted_edges = []

            Deleted_edges = random.sample(Available_edges, k=int(p * G2.number_of_edges()))  # saving the random edges
            G2.remove_edges_from(Deleted_edges)  # delete the edges

            fairness, goodness = fg.compute_fairness_goodness(G2)
            if alpha is None:  # If this method called without alpha param.
                for edge in Deleted_edges:  # all deleted edges for fairness goodness
                    _from = edge[0]
                    _to = edge[1]
                    edge = (*edge, (fairness[_from] * goodness[_to]))  # (from, to, old_weight, new_weight)
                    updated_deleted_edges.append(edge)

            else:  # If this method called with alpha param.
                for edge in Deleted_edges:  # all deleted edges for fairness goodness
                    _from = edge[0]
                    _to = edge[1]
                    # edge = (*edge, ((np.sign(fairness[_from])*(abs(fairness[_from])**alpha))*(np.sign(goodness[_to])*(abs(goodness[_to])**(1-alpha))))) # (from, to, old_weight, new_weight)
                    edge = (*edge, ((np.sign(fairness[_from]) * abs(fairness[_from] ** alpha)) * (
                                np.sign(goodness[_to]) * abs(
                            goodness[_to] ** (1 - alpha)))))  # (from, to, old_weight, new_weight
                    updated_deleted_edges.append(edge)

            temp_rmse.append(calculate_error(updated_deleted_edges, "RMSE"))
        rmse.append(avg(temp_rmse))

    print(rmse)
    return rmse

que = []


def fairness_goodness_convex( a1, a2, alpha):
    return alpha * a1 +(1-alpha) * a2


def fairness_goodness_non_convex( a1, a2, alpha):
    return (np.sign(a1) * abs(a1 ** alpha)) * \
           (np.sign(a2) * abs(a2 ** (1 - alpha)))


def belso_szorzat( a1, a2, alpha):
    return np.array(a1)**alpha @ (np.array(a2)**(1-alpha)).T


def attribute_approx(G, alpha=0.5, v1_method=nx.degree_centrality, v2_method=nx.closeness_centrality,
                     attributum_method=nx.diameter, L=[] ):
    v1 = list(v1_method(G).values())
    v2 = list(v2_method(G).values())
    attributum = attributum_method(G)

    print(v1)
    print(v2)

    #centralitási értékek szozatával köözelítjük a gráf attribútumot.

    print(math.sqrt((belso_szorzat(v1,v2,alpha) - attributum) ** 2)) # RMSE



def lp_constant_p( G, p=0.1, alpha=None, L = None, type=fg.compute_fairness_goodness, convex=False):
    """G: graph
       p: percentage of the edges to be deleted in every iteration exp.: 0.1 = 10%
    """
    # WIP
    # A G referencia lehet h érdemes elöbb másolni a gráfot h az eredetiből ne töröljünk éleket
    if p < 0 or p > 1:
        raise Exception("'p' needs to be between 0 and 1")
    if alpha is not None:
        if alpha < 0 and alpha > 1:
            raise Exception("alpha parameter needs to be between 0 and 1")

    result = {
        'p': p,
        'alpha': alpha
    }
    XPERCENT = math.floor(G.size() * p)  # p percent of the edges
    Available_edges = list(G.edges.data('weight', default=1))# get all edges with weights
    Deleted_edges = []  # deleted edges

    Not_Available_edges = list(nx.complement(G).edges.data('weight', default=0))

    rmse = []

    temp_rmse = []
    non_existent_edges_list = []
    updated_deleted_edges_list = []
    for i in range(25):
        print("p: " + str(p) + " alpha: " + str(alpha))
        G2 = G.copy()
        updated_deleted_edges = []
        non_existent_edges = []

        k = int(p * G2.number_of_edges())

        Deleted_edges = random.sample(Available_edges, k=k)# saving the random edges

        non_existing_edges = None
        if not type == fg.compute_fairness_goodness:
            non_existing_edges = random.sample(Not_Available_edges, k=50)

        G2.remove_edges_from(Deleted_edges)  # delete the edges
        # Innen jön a specifikus rész
        v1, v2 = type(G2,Deleted_edges, non_existing_edges)

        #nx.draw_circular(G2)
        #plt.show()

        #print(v1, v2)
        for edge in Deleted_edges:  # all deleted edges for fairness goodness
            _from = edge[0]
            _to = edge[1]
            # edge = (*edge, ((np.sign(fairness[_from])*(abs(fairness[_from])**alpha))*(np.sign(goodness[_to])*(abs(goodness[_to])**(1-alpha))))) # (from, to, old_weight, new_weight)
            if not convex:
                if type == fg.compute_fairness_goodness:
                    edge = (*edge, (fairness_goodness_non_convex(a1=v1[_from], a2=v2[_to], alpha=alpha)))  # (from, to, old_weight, new_weight
                    updated_deleted_edges.append(edge)
                else:
                    #print(v1)
                    #print(v2)
                    edge = (*edge, (fairness_goodness_non_convex(a1=v1[_from, _to], a2=v2[_from, _to], alpha=alpha)))  # (from, to, old_weight, new_weight
                    updated_deleted_edges.append(edge)
                    #print("updated deleted edges:", updated_deleted_edges)

            else:
                if type == fg.compute_fairness_goodness:
                    edge = (*edge, (fairness_goodness_convex(a1=v1[_from], a2=v2[_to], alpha=alpha)))  # (from, to, old_weight, new_weight
                    updated_deleted_edges.append(edge)
                else:
                    edge = (*edge,(alpha * v1[_from] + (1-alpha) * v2[_to]))
                    updated_deleted_edges.append(edge)

        if not type == fg.compute_fairness_goodness:
            for edge in non_existing_edges:  # all deleted edges for fairness goodness
                _from = edge[0]
                _to = edge[1]
            # edge = (*edge, ((np.sign(fairness[_from])*(abs(fairness[_from])**alpha))*(np.sign(goodness[_to])*(abs(goodness[_to])**(1-alpha))))) # (from, to, old_weight, new_weight)
                if not convex:
                    #print(v1)
                    #print(v2)
                    edge = (*edge, (fairness_goodness_non_convex(a1=v1[_from, _to], a2=v2[_from, _to], alpha=alpha)))  # (from, to, old_weight, new_weight
                    non_existent_edges.append(edge)
                    #print("updated deleted edges:", updated_deleted_edges)

                else:
                    edge = (*edge,(alpha * v1[_from] + (1-alpha) * v2[_to]))
                    non_existent_edges.append(edge)


        updated_deleted_edges_list.append(updated_deleted_edges)
        non_existent_edges_list.append(non_existent_edges)
        #temp_rmse.append(calculate_error(updated_deleted_edges, "RMSE"))
    #rmse.append(avg(temp_rmse))

    """result['rmse'] = rmse
    if L is not None:
        L.append(result)

    print(rmse)"""

    if L is not None:
        if type == fg.compute_fairness_goodness:
            rmse, pearson = main.calculate_errors(updated_deleted_edges_list, non_existent_edges_list,
                                  treshold=G.number_of_nodes() * 0.15, type="fairness goodness")

            result = {
                'rmse': rmse,
                'alpha': alpha,
                'p':p,
                'pearson':pearson
            }

            L.append(result)

        else:
            rmse, auc = main.calculate_errors(updated_deleted_edges_list, non_existent_edges_list,
                                  treshold=G.number_of_nodes() * 0.15, type="")

            result = {
                'rmse': rmse,
                'alpha': alpha,
                'p': p,
                'auc': auc
            }

            L.append(result)


    return updated_deleted_edges_list, non_existent_edges_list

def run_paralell_lp(G, func, process_number = 4):
    alpha_v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    p_v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rmse = {
        0.1: {},
        0.2: {},
        0.3: {},
        0.4: {},
        0.5: {},
        0.6: {},
        0.7: {},
        0.8: {},
        0.9: {},
    }

    auc = {
        0.1: {},
        0.2: {},
        0.3: {},
        0.4: {},
        0.5: {},
        0.6: {},
        0.7: {},
        0.8: {},
        0.9: {},
    }

    pearson = {
        0.1: {},
        0.2: {},
        0.3: {},
        0.4: {},
        0.5: {},
        0.6: {},
        0.7: {},
        0.8: {},
        0.9: {},
    }

    with multiprocessing.Manager() as manager: # start parralell processes
        L = manager.list() # list to save the Process results
        processes = []
        pool = multiprocessing.Pool(processes=process_number)
        for alpha_value in alpha_v:
            for p_value in p_v:
                result = pool.apply_async(lp_constant_p, [G, p_value, alpha_value, L, func])
                # p.start() # start each process
                # processes.append(p)
                print(result)
        pool.close()
        pool.join()
        for p in processes:
            p.join()  # wait for all process to be terminated
        print(L)
        for i in L:
            tmp = i
            print(tmp)
            rmse[tmp['alpha']][tmp['p']] = tmp['rmse']
            try:
                auc[tmp['alpha']][tmp['p']] = tmp['auc']
            except:
                pass

            try:
                pearson[tmp['alpha']][tmp['p']] = tmp['pearson']
            except:
                pass

        print(rmse)
        print(auc)
        print(pearson)
        
        return rmse, auc, pearson


def plotting(file):
    results_strkeys = None
    with open(file, 'r') as fp:
        results_strkeys = json.loads(fp.read())
    results = {
        0.1: {},
        0.2: {},
        0.3: {},
        0.4: {},
        0.5: {},
        0.6: {},
        0.7: {},
        0.8: {},
        0.9: {}
    }

    alpha_plot_index = {
        0.1: [0, 0],
        0.2: [0, 1],
        0.3: [0, 2],
        0.4: [1, 0],
        0.5: [1, 1],
        0.6: [1, 2],
        0.7: [2, 0],
        0.8: [2, 1],
        0.9: [2, 2]
    }

    alpha_v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    p_v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fig, axs = plt.plots(3, 3, sharex=True, sharey=True )



    #fig = plt.figure()
    #gs = fig.add_gridspec(3, 3, hspace=5)
    #axs = gs.subplots(sharex=True, sharey=True)

    for alpha in alpha_v:
        for p in p_v:
            results[alpha][p] = results_strkeys[str(alpha)][str(p)]
        print("Alpha value: {}".format(alpha))

        firstIndex = alpha_plot_index[alpha][0]
        secondIndex = alpha_plot_index[alpha][1]
        print("row: {} column: {}".format(firstIndex, secondIndex))

        axs[firstIndex, secondIndex].plot(p_v, list(results[alpha].values()))
        #axs[firstIndex, secondIndex].set_title(f"Alpha value: {alpha}")


