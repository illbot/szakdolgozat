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


def calculate_error(deleted_edges, Type):
    if Type == "RMSE":
        old_w = []
        predicted_w = []

        for edge in deleted_edges:
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


def lp_constant_p( G, p=0.1, alpha=None, L = None):
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
    Available_edges = list(G.edges.data('weight', default=0))  # get all edges with weights
    Deleted_edges = []  # deleted edges

    rmse = []

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

    result['rmse'] = rmse
    if L is not None:
        L.append(result)

    print(rmse)

    return rmse

def draw_plots(G, process_number = 4):
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
    with multiprocessing.Manager() as manager: # start parralell processes
        L = manager.list() # list to save the Process results
        processes = []
        pool = multiprocessing.Pool(processes=process_number)
        for alpha_value in alpha_v:
            for p_value in p_v:
                result = pool.apply_async(lp_constant_p, [G,p_value,alpha_value, L])
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
            rmse[tmp['alpha']][tmp['p']] = tmp['rmse'][0]
        print(rmse)
        
        return rmse


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
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True )
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


