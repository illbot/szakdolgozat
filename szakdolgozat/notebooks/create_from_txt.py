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


def alpha(url):
    G = nx.DiGraph()
    data = pd.read_csv(url, sep=",", header=None)
    data.columns = ["source", "target", "rating"]

    for index, row in data.iterrows():
        G.add_node(row['source'].item())

    for index, row in data.iterrows():
        G.add_edge(row['source'].item(), row['target'].item(), weight=float(row['rating'].item()))
    return G

def otc(url):
    G = nx.DiGraph()
    data = pd.read_csv(url, sep=",", header=None)
    data.columns = ["source", "target", "rating", "time"]

    for index, row in data.iterrows():
        G.add_node(row['source'].item())

    for index, row in data.iterrows():
        G.add_edge(row['source'].item(), row['target'].item(), weight=float(row['rating'].item()))
    return G



def facebook(url):
    G = nx.Graph()
    data = pd.read_csv(url, sep=" ", header=None)
    data.columns = ["from", "to"]
    for index, row in data.iterrows():
        G.add_node(row['from'].item())

    for index, row in data.iterrows():
        G.add_edge(row['from'].item(), row['to'].item())
    return G