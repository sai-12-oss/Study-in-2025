import re

class CFG:
    def __init__(self, productions):
        self.productions = self.parse_productions(productions)

    def parse_productions(self, productions):
        production_dict = {}
        if not productions.strip():  # Handle empty input
            return production_dict
        for production in productions.split(';'):
            production = production.strip()
            if not production or '->' not in production:  # Skip empty or invalid productions
                continue
            head, body = production.split('->')
            head = head.strip()
            bodies1 = body.split('|')
            bodies = []
            for b in bodies1:
                b = b.strip()
                if b:  # Skip empty bodies
                    bodies.append(list(b))  # Split into individual characters
            if head and bodies:  # Only add valid productions
                if head not in production_dict:
                    production_dict[head] = []
                production_dict[head].extend(bodies)
        return production_dict

class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_edge(self, source, target, label):
        if source not in self.adjacency_list:
            self.adjacency_list[source] = []
        self.adjacency_list[source].append((target, label))

    def get_edges(self):
        return self.adjacency_list