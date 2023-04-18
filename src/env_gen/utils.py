import numpy as np


def permute_graph(adj_matrix, rng):
    # Puts the nodes in the graph in a random order
    n_states = adj_matrix.shape[0]
    permutation = rng.permutation(n_states)
    return adj_matrix[permutation, :][:, permutation]


def dfs(graph, vertex, visited):
    visited[vertex] = True
    for neighbor, edge in enumerate(graph[vertex]):
        if edge and not visited[neighbor]:
            dfs(graph, neighbor, visited) 


def is_strongly_connected(graph):
    # Graph is adj. matrix of the shape [n_nodes, n_nodes]
    n = len(graph)
    # Step 1: Perform DFS on the original graph
    visited = np.zeros(n, dtype=bool)
    dfs(graph, 0, visited)
    if not all(visited):
        return False
    # Step 2: Transpose the graph and perform DFS again
    graph_transposed = graph.T
    visited = np.zeros(n, dtype=bool)
    dfs(graph_transposed, 0, visited)
    return all(visited)


def random_spanning_tree(n, rng):
    # Initialize spanning tree
    spanning_tree = np.zeros((n, n), dtype=int)
    visited = set()
    # Start from a random node
    current = rng.choice(range(n))
    visited.add(current)
    # Aldous-Broder algorithm
    while len(visited) < n:
        next_node = rng.choice([i for i in range(n) if i != current])
        if next_node not in visited:
            visited.add(next_node)
            spanning_tree[current, next_node] = 1
            spanning_tree[next_node, current] = 1
        current = next_node
    return spanning_tree


def visualize_policy():
    pass