from helpers import CFG
from helpers import Graph

def check_reachability(cfg, graph, start_vertex, end_vertex):
    # step-1 : get all the productions and arrange them into terminals and non-terminals 
    terminals = {}
    non_terminals = {}
    for nt,productions in cfg.productions.items():
        for production in productions:
            if len(production) == 1:
                label = production[0]
                if label not in terminals:
                    terminals[label] = set()
                terminals[label].add(nt)
            elif len(production) == 2:
                X,Y = production[0],production[1]
                if (X,Y) not in non_terminals:
                    non_terminals[(X,Y)] = set()
                non_terminals[(X,Y)].add(nt)
    # print(f"Terminals: {terminals}")
    # print(f"Non-Terminals: {non_terminals}")
    # step-2 : Create the Solver table 
    Solvable = {}
    nodes = set()
    edges = graph.get_edges()
    for source in edges:
        nodes.add(source)
        for distance,label in edges[source]:
            nodes.add(distance)
            if (source,distance) not in Solvable:
                Solvable[(source,distance)] = set()
            # step-3 : Add non-termianls for single edges 
            if label in terminals:
                Solvable[(source,distance)].update(terminals[label])
    # print("Initial Solvable Table:")
    # for (u, v), nt_set in sorted(Solvable.items()):
    #     print(f"({u}, {v}): {sorted(nt_set)}")
    # step-4 : Updating non-terminal productions iteratively 
    changed = True 
    # iteration = 1
    while changed:
        changed = False 
        # for each pair of nodes (u,v) and (v,w)
        for u in nodes:
            for w in nodes:
                for v in nodes:
                    if (u,w) in Solvable and (w,v) in Solvable:
                        X_set = Solvable[(u,w)]
                        Y_set = Solvable[(w,v)]
                        for X in X_set:
                            for Y in Y_set:
                                if (X,Y) in non_terminals:
                                    Z_set = non_terminals[(X,Y)]
                                    if (u,v) not in Solvable:
                                        Solvable[(u,v)] = set()
                                        # Add new non-terminals to (u,v)
                                    new_additions = Z_set - Solvable[(u,v)]
                                    if new_additions:
                                        Solvable[(u,v)].update(new_additions)
                                        changed = True
        # if changed:
        #     print(f"\nSolvable Table after Iteration {iteration}:")
        #     for (u, v), nt_set in sorted(Solvable.items()):
        #         print(f"({u}, {v}): {sorted(nt_set)}")
        #     iteration += 1
    # step-5 : Check if the start symbol 'S' can generate path form start to end 
    result = (start_vertex, end_vertex) in Solvable and 'S' in Solvable[(start_vertex, end_vertex)]
    # print(f"\nFinal Check: ({start_vertex}, {end_vertex}) has 'S': {result}")
    return result

    return False

def read_input(file_path):
    with open(file_path, 'r') as file:
        num_inputs = int(file.readline().strip())
        inputs = []
        for _ in range(num_inputs):
            cfg_productions = file.readline().strip()
            graph_data = file.readline().strip()
            start_vertex = file.readline().strip()
            end_vertex = file.readline().strip()
            inputs.append((cfg_productions, graph_data, start_vertex, end_vertex))
        return inputs
# print((read_input('input.txt')))
def write_output(file_path, results):
    with open(file_path, 'w') as file:
        for result in results:
            file.write(result + '\n')

def main(input_file, output_file):
    inputs = read_input(input_file)
    results = []
    for cfg_productions, graph_data, start_vertex, end_vertex in inputs:
        cfg = CFG(cfg_productions)
        graph = Graph()
        edge_data = graph_data.split(' ')
        for edge in edge_data:
            src = edge[0]
            dst = edge[1]
            label = edge[3]
            graph.add_edge(src, dst, label)
        print(cfg.parse_productions(cfg_productions))
        reachable = check_reachability(cfg, graph, start_vertex, end_vertex)
        results.append('Yes' if reachable else 'No')
    
    write_output(output_file, results)

if __name__ == "__main__":
    input_file = 'input.txt'
    output_file = 'output.txt'
    main(input_file, output_file)

