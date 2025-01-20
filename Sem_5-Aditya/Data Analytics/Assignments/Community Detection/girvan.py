import numpy as np
import networkx as nx
from collections import deque, defaultdict
from multiprocessing import Pool, cpu_count


# creating graph from edges
def create_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)

    return G


# finding connected components through BFS
def connected_components(G):
    components = []
    visited = set()

    for v in G.nodes():
        if v in visited:
            continue

        component = [v]
        queue = deque([v])
        visited.add(v)

        while queue:
            v = queue.popleft()

            for w in G[v]:
                if w not in visited:
                    component.append(w)
                    queue.append(w)
                    visited.add(w)

        components.append(component)

    return components


# finding community id for each node from partition
def graph_partition(components):
    # setting id to min of the component
    community_id = [min(i) for i in components]
    graph_partition = []

    for i in range(len(components)):
        for j in components[i]:
            graph_partition.append([j, community_id[i]])
    
    graph_partition = np.array(graph_partition)
    graph_partition = graph_partition[graph_partition[:,0].argsort()][:,1]

    return graph_partition


# finding modularity of the graph
def modularity(G, components):
    m = G.number_of_edges()
    Q = 0

    for component in components:
        for i in component:
            for j in G[i]:
                if j in component:
                    Q += 1
        Q -= sum([G.degree(k) for k in component])**2 / (2*m)  
    Q /= 2*m

    return Q


# finding edge betweenness for a node
def girvan_newman_edge_betweenness(G, S):
    edge_betweenness = defaultdict(float)

    for s in S:
        # weight is the number of shortest paths from the node to s
        weight = {i: 0 for i in G.nodes()}
        # parent list for each node
        parent = {i: [] for i in G.nodes()}
        # depth of the node in the BFS tree
        depth = {i: -1 for i in G.nodes()}
        
        queue = deque([s])

        # order of finding the nodes in BFS
        discovery = []

        weight[s] = 1
        depth[s] = 0

        while queue:
            v = queue.popleft()
            discovery.append(v)

            for w in G[v]:
                if depth[w] < 0:
                    queue.append(w)
                    depth[w] = depth[v] + 1
                    
                if depth[w] == depth[v] + 1:
                    weight[w] += weight[v]
                    parent[w].append(v)

        # credit for each node
        credit = {i: 1 for i in G.nodes()}
        
        while discovery:
            v = discovery.pop()

            # contribution of the node to the edge betweenness
            for w in parent[v]:
                contribution = (weight[w] / weight[v]) * (credit[v])
                edge = (v, w) if v < w else (w, v)
                edge_betweenness[edge] += contribution
                credit[w] += contribution

    return edge_betweenness


# running Girvan-Newman till a split is found
def girvan_newman_one_level(edges=None, G=None):
    if edges is not None:
        G = create_graph(edges)
        
    initial_components = connected_components(G)
    components = initial_components
    
    nodes = np.array(G.nodes())
    # shuffling to prevent high load on a single core
    np.random.shuffle(nodes)
    node_list = np.array_split(nodes, cpu_count()-1)
    
    while len(components) == len(initial_components):
        # parellelizing the edge betweenness calculation between cpu cores
        with Pool(cpu_count()-1) as pool:
            betweenness = pool.starmap(girvan_newman_edge_betweenness, [(G, S) for S in node_list])

        edge_betweenness = defaultdict(float)

        for b in betweenness:
            for edge, value in b.items():
                edge_betweenness[edge] += value/2

        edge_betweenness = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
        max_edge = edge_betweenness[0][0]

        G.remove_edges_from([max_edge])
        edges = np.array(G.edges())
        components = connected_components(G)

    partition = graph_partition(components)
    
    return partition, G


# full Girvan-Newman algorithm
def girvan_newman(edges):
    G = create_graph(edges)
    G_new = G.copy()
    
    modularity_initial = modularity(G, connected_components(G))
    modularity_final = modularity_initial
    
    community_mat = []
    partition = graph_partition(connected_components(G))

    # running till modularity increases
    while modularity_final >= modularity_initial:
        community_mat.append(partition)
        modularity_initial = modularity_final

        partition, G_new = girvan_newman_one_level(G=G_new)
        modularity_final = modularity(G, connected_components(G_new))

    community_mat = np.transpose(np.array(community_mat))

    return community_mat