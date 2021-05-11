import pandas as pd
import networkx as nx
import time
import matplotlib.pyplot as plt


def from_txt(url):
    G = nx.Graph()
    data = pd.read_csv(url, sep=",", header=None)
    data.columns = ["from", "to", "rating"]
    for index, row in data.iterrows():
        G.add_node(row['from'].item())

    for index, row in data.iterrows():
        G.add_edge(row['from'].item(), row['to'].item())
    return G


def from_csv(url):
    G = nx.DiGraph()
    data = pd.read_csv(url, sep=",", header=None)
    data.columns = ["source", "target", "rating"]

    for index, row in data.iterrows():
        G.add_node(row['source'].item())

    for index, row in data.iterrows():
        G.add_edge(row['source'].item(), row['target'].item(), weight=float(row['rating'].item()))
    return G