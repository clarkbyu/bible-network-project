# app.py
"""
@author: Clark Brown
@date: 15 April 2020
"""

import math
import networkx as nx
from networkx import algorithms as alg
import csv
import numpy as np
import collections
from matplotlib import pyplot as plt
from scipy import stats
from scipy import optimize

class NeworkDataDriver:
    def __init__(self, data_file='bible_names.csv'):
        self.graph = nx.Graph()

        with open(data_file, 'r') as infile:
            rows = list(csv.reader(infile))[1:]
            for name1, name2, freq in rows:
                self.graph.add_edge(name1, name2, weight=int(freq))
    
    def degree_sequence(self):
        return sorted([d for n, d in self.graph.degree()], reverse=True)

    def degree_distribution(self):
        degree_count = collections.Counter(self.degree_sequence())
        deg, cnt = zip(*degree_count.items())

        # Regression
        powerlaw = lambda x, m, a : a * (x ** m)
        popt, pcov = optimize.curve_fit(powerlaw, deg, cnt, maxfev=2000)

        # Plot distribution
        plt.yscale("symlog")
        plt.xscale("symlog")
        plt.scatter(deg, cnt)
        dom = np.linspace(2, 80)
        plt.plot(dom, powerlaw(dom, *popt), 'g--')

        plt.title("Degree Distribution")
        plt.ylabel("Count")
        plt.xlabel("Degree")

        plt.show()

    def connected_components(self, largest_first=True):
        return sorted(nx.connected_components(self.graph), key=len, reverse=largest_first)

    def giant_component(self):
        gc = self.graph.subgraph(self.connected_components()[0])
        proportion = gc.number_of_nodes() / self.graph.number_of_nodes()
        return gc, proportion

    def component_size_distribution(self):
        ccs = self.connected_components()
        cc_sizes = [self.graph.subgraph(cc).number_of_nodes() for cc in ccs]
        size_count = collections.Counter(cc_sizes)
        size, cnt = zip(*size_count.items())
        print(size)
        # Regression
        powerlaw = lambda x, m, a : a * (x ** m)
        popt, pcov = optimize.curve_fit(powerlaw, size, cnt, maxfev=2000)

        plt.yscale("symlog")
        plt.xscale("symlog")
        plt.scatter(size, cnt)
        plt.plot(size, powerlaw(size, *popt), 'g--')
        plt.xlim(2, 2000)
        plt.title("Component Distribution")
        plt.ylabel("Count")
        plt.xlabel("Component Size")

        plt.show()

    def get_k_cores(self, k=10):
        cliques = alg.community.k_clique_communities(self.graph, k)
        return sorted(map(sorted, cliques))

    def k_components(self):
        print("Don't. Just don't.")

    def get_ranks(self, d):
        """Construct a sorted list of labels based on the PageRank vector.

        Parameters:
            d (dict(str -> float)): a dictionary mapping labels to PageRank values.

        Returns:
            (list) the keys of d, sorted by PageRank value from greatest to least.
        """
        keys = sorted(d.keys())
        values = [d[i] for i in keys]

        # Order from lowest to highest
        order = np.argsort(values)[::-1]
        ordered = [keys[order[i]] for i in range(len(keys))]

        return ordered

    def betweenness_centrality(self, k=10):
        p = nx.betweenness_centrality(self.graph)
        return self.get_ranks(p)[:k]

    def closeness_centrality(self, k=10, distance=None):
        p = nx.closeness_centrality(self.graph, distance=distance)
        return self.get_ranks(p)[:k]

    def degree_centrality(self, k=10):
        p = nx.degree_centrality(self.graph)
        return self.get_ranks(p)[:k]

    def eigenvector_centrality(self, k=10):
        p = nx.eigenvector_centrality(self.graph)
        return self.get_ranks(p)[:k]

    def katz_centrality(self, alpha=0.1, beta=1.0, k=10):
        p = nx.katz_centrality(self.graph, alpha=alpha, beta=beta)
        return self.get_ranks(p)[:k]

    def hits(self, k=10):
        hubs, authorities = nx.hits(self.graph)
        return self.get_ranks(hubs)[:k], self.get_ranks(authorities)[:k]

    def pagerank(self, alpha=0.85, k=10):
        p = nx.pagerank(self.graph, alpha=alpha)
        return self.get_ranks(p)[:k]

    def edge_betweeness_centrality(self, k=10):
        p = nx.edge_betweeness_centrality(self.graph)
        return self.get_ranks(p)[:k]

    def is_planar(self):
        return alg.planarity.check_planarity(self.graph, counterexample=False)[0]

    def is_connected(self):
        return alg.components.is_connected(self.graph)

    def get_diameter(self):
        if self.is_connected():
            return alg.distance_measures.diameter(self.graph)
        else:
            return alg.distance_measures.diameter(self.giant_component()[0])

    def clustering(self):
        return len(alg.cluster.triangles(self.graph)), alg.cluster.transitivity(self.graph), alg.cluster.average_clustering(self.graph)

    def small_world(self):
        return alg.smallworld.sigma(self.graph)

    def density(self):
        return np.mean(self.degree_sequence()) / (self.graph.number_of_nodes() - 1)

    def assortativity(self):
        return nx.degree_assortativity_coefficient(self.graph), nx.degree_assortativity_coefficient(self.graph, weight='weight')

    def get_node_degree(self, node):
        return nx.degree(self.graph, node)

    def get_async_fluid_communities(self, k):
        comms = list(alg.community.asyn_fluidc(self.giant_component()[0], k))
        return [sorted(com) for com in comms]

    def __str__(self):
        nodes = "Number of nodes: " + str(self.graph.number_of_nodes())
        edges = "\nNumber of edges: " + str(self.graph.number_of_edges())
        loops = "\nNumber of self-loops: " + str(self.graph.number_of_selfloops())
        return nodes + edges + loops

""" USAGE:
if __name__ == '__main__':
    driver = NeworkDataDriver()
"""