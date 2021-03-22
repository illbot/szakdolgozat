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

def missing_link_algorithm(G,x,y, alpha):
    neighbors_of_x = list(nx.neighbors(G, x))
    neighbors_of_y = list(nx.neighbors(G, y))
    common_elements = len(intersection(neighbors_of_x, neighbors_of_y))

    shortest_distance = len(nx.shortest_path(G, x,y))

    closeness = G.number_of_nodes() / shortest_distance

    result = alpha * common_elements + (1-alpha) * closeness

    return result

'''
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
print(missing_link_algorithm(G,3,2,0.1))'''
