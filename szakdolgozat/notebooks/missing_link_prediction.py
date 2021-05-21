import networkx as nx
import numpy as np
import math
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import threading
import multiprocessing
import concurrent.futures
import queue
import time
from multiprocessing.managers import SharedMemoryManager
import json

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def missing_link_algorithm(G, deleted_edges, non_existing_edges):
    m1 = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    m2 = np.zeros((G.number_of_nodes(), G.number_of_nodes()))

    #print("non-existing edges", non_existing_edges)

    #for n in G.nodes:
    #    print(nx.is_isolate(G,n))

    for edge in non_existing_edges:
        neighbors_of_x = list(nx.neighbors(G, edge[0]))
        neighbors_of_y = list(nx.neighbors(G, edge[1]))
        common_elements = len(intersection(neighbors_of_x, neighbors_of_y))

        shortest_distance = None
        try:
            shortest_distance = len(nx.shortest_path(G, edge[0], edge[1]))
        except Exception as e:
            shortest_distance = -1

        closeness = G.number_of_nodes() / shortest_distance

        m1[int(edge[0]), int(edge[1])] = common_elements
        m2[int(edge[0]), int(edge[1])] = closeness


    for edge in deleted_edges:
        neighbors_of_x = list(nx.neighbors(G, edge[0]))
        neighbors_of_y = list(nx.neighbors(G, edge[1]))
        common_elements = len(intersection(neighbors_of_x, neighbors_of_y))

        shortest_distance = None
        try:
            shortest_distance = len(nx.shortest_path(G, edge[0],edge[1]))
        except Exception as e:
            shortest_distance = -1

        closeness = G.number_of_nodes() / shortest_distance

        m1[int(edge[0]), int(edge[1])] = common_elements
        m2[int(edge[0]), int(edge[1])] = closeness

    return m1, m2




def missing_link_pred_p(G, p=0.1, alpha=None, L = None):
    """ G: graph
        p: percentage of the edges to be deleted in every iteration exp.: 0.1 = 10%
        alpha: [0 - 1] number
    """
    if p < 0 or p > 1:
        raise Exception("'p' needs to be between 0 and 1")
    if alpha is not None:
        if alpha < 0 and alpha > 1:
            raise Exception("alpha parameter needs to be between 0 and 1")

    Available_edges = list(G.edges.data())

    #for i in range(25):
    G2 = G.copy()
    updated_deleted_edges = []

    Deleted_edges = random.sample(Available_edges, k=int(p * G2.number_of_edges()))  # saving the random edges
    G2.remove_edges_from(Deleted_edges)  # delete the edges

    result = missing_link_algorithm(G2, )


"""
# TEST
G = nx.Graph()
nx.add_path(G,[0,1,2,3])
nx.add_path(G,[1,4,3])

asd = list(nx.neighbors(G,1))

for i in asd:
    print(i)
print(type(asd))
print(asd)

nx.shortest_path(G,0,4)
print("shotest path: ",nx.shortest_path(G,0,4))
print(type(nx.shortest_path(G,0,4)))

print("number of nodes: ",G.number_of_nodes())

print("Algorithm test:")
print(missing_link_algorithm(G,1,2,0.1))
print(missing_link_algorithm(G,0,3,0.1))
print(missing_link_algorithm(G,0,4,0.1))
print(missing_link_algorithm(G,3,2,0.1))"""
