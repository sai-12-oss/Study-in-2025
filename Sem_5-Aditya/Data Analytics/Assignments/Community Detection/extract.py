import numpy as np
import networkx as nx

# creating graph from edges
def create_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)

    return G


# removing duplicate edges
def edge_clean(edges):
    # sort the edges
    edges = edges[np.lexsort((edges[:,1],edges[:,0]))]

    duplicates = []
    for i in range(len(edges)):
        if i in duplicates:
            continue
        
        # finding first occurence of the edge
        a = np.searchsorted(edges[:,0], edges[i][1], side='left')
        if a is not None:
            b = np.searchsorted(edges[:,0], edges[i][1], side='right')
            c = np.searchsorted(edges[a:b,1], edges[i][0], side='left')

            # if the edge is found
            if c < (b-a) and edges[a+c][1] == edges[i][0]:
                duplicates.append(a+c)

    # removing duplicates
    edges = np.delete(edges, duplicates, axis=0)

    return edges


# importing wiki vote data
def import_wiki_vote_data(path="data/Wiki-Vote.txt"):
    f = open(path, "r")
    lines = f.readlines()
    lines = lines[4:]
    f.close()

    edges = np.zeros((len(lines), 2), dtype=int)
    for i in range(len(lines)):
        edge = lines[i].strip().split("\t")
        edges[i] = [int(x) for x in edge]
        
    edges = edge_clean(edges)

    return edges


# importing lastfm asia data
def import_lastfm_asia_data(path="data/lastfm_asia_edges.csv"):
    f = open(path, "r")
    lines = f.readlines()
    lines = lines[1:]
    f.close()

    edges = np.zeros((len(lines), 2), dtype=int)
    for i in range(len(lines)):
        edge = lines[i].strip().split(",")
        edges[i] = [int(x) for x in edge]
        
    edges = edge_clean(edges)
 
    return edges