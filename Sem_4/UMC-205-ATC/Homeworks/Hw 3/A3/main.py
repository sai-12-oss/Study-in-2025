from helpers import CFG
from helpers import Graph

def check_reachability(cfg, graph, start_vertex, end_vertex):
    # step-1 : get all the productions and arrange them into terminal_pairs and non-terminal_pairs 
    terminal_pairs = {}
    non_terminal_pairs = {}
    for nt,productions in cfg.productions.items():
        for production in productions:
            if len(production) == 1:
                label = production[0]
                if label not in terminal_pairs:
                    terminal_pairs[label] = set()
                terminal_pairs[label].add(nt)
            elif len(production) == 2:
                X,Y = production[0],production[1]
                if (X,Y) not in non_terminal_pairs:
                    non_terminal_pairs[(X,Y)] = set()
                non_terminal_pairs[(X,Y)].add(nt)
    # print(f"Terminal_pairs: {terminal_pairs}")
    # print(f"Non-terminal_pairs: {non_terminal-pairs}")
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
            if label in terminal_pairs:
                Solvable[(source,distance)].update(terminal_pairs[label])
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
                        for X in X_set.copy(): # as we dont want it to get updated everytime 
                            for Y in Y_set.copy():
                                if (X,Y) in non_terminal_pairs:
                                    Z_set = non_terminal_pairs[(X,Y)]
                                    if (u,v) not in Solvable:
                                        Solvable[(u,v)] = set()
                                        # Add new non-terminal_pairs to (u,v)
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
        if graph_data.strip():  # Ensure non-empty after stripping
            edge_data = [e for e in graph_data.split(' ') if e]  # Filter out empty strings
            for edge in edge_data:
                src = edge[0]
                dst = edge[1]
                label = edge[3]
                graph.add_edge(src, dst, label)
        reachable = check_reachability(cfg, graph, start_vertex, end_vertex)
        results.append('YES' if reachable else 'NO')
    write_output(output_file, results)

if __name__ == "__main__":
    input_file = 'input.txt'
    output_file = 'output.txt'
    main(input_file, output_file)

