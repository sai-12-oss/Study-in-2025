import numpy as np
import networkx as nx
from collections import defaultdict


# creating graph from edges
def create_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)

    return G


# finding community id for each node from partition
def graph_partition(components):
    community_id = [min(i) for i in components]
    graph_partition = []

    for i in range(len(components)):
        for j in components[i]:
            graph_partition.append([j, community_id[i]])
    
    graph_partition = np.array(graph_partition)
    graph_partition = graph_partition[graph_partition[:,0].argsort()][:,1]

    return graph_partition


# finding number of neighbours in each community
def neighbour_community(G, C, node):
    neighbours = defaultdict(int)
    
    for i in G[node]:
        neighbours[C[i]] += 1
    neighbours[C[node]] += 0

    return neighbours


# one run of louvain algorithm
def louvain(G, C, community_weight, threshold=0.000001):
    m = G.number_of_edges()
    update = False

    nodes = np.array(G.nodes())
    # shuffle the nodes to avoid bias
    np.random.shuffle(nodes)
    for i in nodes:
        x = C[i]
        neighbours = neighbour_community(G, C, i)
        
        deg_i = G.degree(i)
        deg_i_x = neighbours.pop(x)

        max_change = 0
        best_community = x
        
        for y in neighbours.keys():
            deg_i_y = neighbours[y]
            # change in modularity
            change = (2*(deg_i_y - deg_i_x) - deg_i*(community_weight[y] - community_weight[x])/m)/(2*m)
            
            if change > max_change + threshold:
                max_change = change
                best_community = y

        if best_community != x:
            update = True
            C[i] = best_community
            community_weight[best_community] += deg_i
            community_weight[x] -= deg_i

            if community_weight[x] == 0:
                del community_weight[x]

    return C, community_weight, update


# full phase 1 of louvain algorithm
def louvain_one_iter(edges):
    G = create_graph(edges)
    C = {i: i for i in G.nodes()}
    community_weight = {i: G.degree(i) for i in G.nodes()}

    update = True
    # run till there are no changes in the graph
    while update:
        C, community_weight, update = louvain(G, C, community_weight)

    components = {i : [] for i in C.values()}
    for node in C.keys():
        components[C[node]].append(node)

    components = [i for i in components.values()]
    partition = graph_partition(components)

    return partition