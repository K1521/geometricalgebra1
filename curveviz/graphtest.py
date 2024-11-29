import heapq
import time
import random

class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v, d):
        self.adj_list.setdefault(u, []).append((v, d))
        self.adj_list.setdefault(v, []).append((u, d))  # For undirected graph

    def getnextnodes_with_check(self, u):
        heap = [(0, u)]
        visited = set()
        while heap:
            d, v = heapq.heappop(heap)
            if v in visited:
                continue
            visited.add(v)
            yield v, d
            for n, dvn in self.adj_list[v]:
                if n not in visited:
                    heapq.heappush(heap, (d + dvn, n))

    def getnextnodes_without_check(self, u):
        heap = [(0, u)]
        visited = set()
        while heap:
            d, v = heapq.heappop(heap)
            if v in visited:
                continue
            visited.add(v)
            yield v, d
            for n, dvn in self.adj_list[v]:
                heapq.heappush(heap, (d + dvn, n))


def test_performance(graph, start_node):
    # Test with the check for visited nodes
    start_time = time.time()
    for i in range(100):
        for _ in graph.getnextnodes_with_check(start_node):
            pass
    with_check_time = time.time() - start_time

    # Test without the check for visited nodes
    start_time = time.time()
    for i in range(100):
        for _ in graph.getnextnodes_without_check(start_node):
            pass
    without_check_time = time.time() - start_time

    return with_check_time, without_check_time


# Create a large graph for testing with random edges
def create_large_graph(num_nodes, num_edges):
    graph = Graph()
    nodes = [f'Node_{i}' for i in range(num_nodes)]
    for _ in range(num_edges):
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v:  # Ensure no self-loops
            d = random.randint(1, 100)  # Random weights between 1 and 100
            graph.add_edge(u, v, d)
    return graph

# Example usage
large_graph = create_large_graph(1000, 50000)  # 1000 nodes, 5000 edges
with_check_time, without_check_time = test_performance(large_graph, 'Node_0')

print(f"With check: {with_check_time:.4f} seconds")
print(f"Without check: {without_check_time:.4f} seconds")
