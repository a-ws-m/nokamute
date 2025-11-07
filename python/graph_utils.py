"""
Graph utility functions for converting board states to NetworkX graphs
and computing graph hashes for position equivalence.
"""

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data


def board_to_networkx(node_features, edge_index):
    """
    Convert a board state graph (as node features and edge index) to a NetworkX graph.
    
    Args:
        node_features: List of lists containing node feature vectors
        edge_index: List of two lists [sources, targets] representing edges
        
    Returns:
        NetworkX graph with node attributes
    """
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index_tensor)
    
    # Convert to NetworkX - store node features individually per node
    G = to_networkx(data, node_attrs=['x'], to_undirected=True)
    
    # Ensure each node has its feature vector stored correctly
    # to_networkx stores the feature for each node separately
    for node_id in G.nodes():
        if 'x' not in G.nodes[node_id]:
            # Fallback: manually assign features if not present
            G.nodes[node_id]['x'] = x[node_id]
    
    return G


def graph_hash(node_features, edge_index):
    """
    Compute a hash for a graph using the Weisfeiler-Lehman algorithm.
    
    This provides a canonical hash that is invariant to node relabeling,
    making it more robust than zobrist hashing for graph isomorphism.
    
    Args:
        node_features: List of lists containing node feature vectors
        edge_index: List of two lists [sources, targets] representing edges
        
    Returns:
        String hash representing the graph structure and features
    """
    # Convert to NetworkX
    G = board_to_networkx(node_features, edge_index)
    
    # Convert node features to hashable strings for WL hashing
    # The 'x' attribute contains the feature tensors
    for node_id, data in G.nodes(data=True):
        # Convert feature vector to hashable string
        # Handle both tensor and list formats
        x_data = data['x']
        if hasattr(x_data, 'numpy'):
            # It's a tensor
            features = tuple(x_data.numpy().round(4))
        else:
            # It's already a list or array
            features = tuple(np.round(x_data, 4))
        # Store as hashable string representation
        G.nodes[node_id]['label'] = str(features)
    
    # Compute Weisfeiler-Lehman hash using the node labels
    # This ensures that nodes with different features get different hashes
    wl_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr='label', edge_attr=None)
    
    return wl_hash


def networkx_to_pyg(G):
    """
    Convert a NetworkX graph back to PyTorch Geometric format.
    
    Args:
        G: NetworkX graph with node attributes
        
    Returns:
        Tuple of (node_features, edge_index) as Python lists
    """
    # Check if the graph has the 'x' attribute on nodes
    if len(G.nodes()) == 0:
        return [], [[], []]
    
    # Verify that nodes have the 'x' attribute
    first_node = list(G.nodes())[0]
    if 'x' not in G.nodes[first_node]:
        raise ValueError(f"NetworkX graph nodes missing 'x' attribute. Node {first_node} has attributes: {list(G.nodes[first_node].keys())}")
    
    try:
        data = from_networkx(G, group_node_attrs=['x'])
    except KeyError as e:
        # If 'x' attribute is not found, try to extract features manually
        print(f"Warning: Failed to convert NetworkX graph using from_networkx: {e}")
        print(f"Attempting manual conversion...")
        
        # Manual conversion
        node_features = []
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        for node in sorted(G.nodes()):
            node_data = G.nodes[node]
            if 'x' in node_data:
                x = node_data['x']
                # Convert tensor to list if needed
                if hasattr(x, 'numpy'):
                    x = x.numpy().tolist()
                elif hasattr(x, 'tolist'):
                    x = x.tolist()
                node_features.append(x)
            else:
                raise ValueError(f"Node {node} missing 'x' attribute")
        
        # Extract edges
        edge_index = [[], []]
        for u, v in G.edges():
            edge_index[0].append(node_mapping[u])
            edge_index[1].append(node_mapping[v])
            # Add reverse edge for undirected graph
            edge_index[0].append(node_mapping[v])
            edge_index[1].append(node_mapping[u])
        
        return node_features, edge_index
    
    node_features = data.x.numpy().tolist()
    edge_index = data.edge_index.numpy().tolist()
    
    return node_features, edge_index


if __name__ == "__main__":
    # Test the utility functions
    print("Testing graph utilities...")
    
    # Create a simple test graph
    # Features: [color, bug_onehot (9), height, current_player]
    node_features = [
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # White's turn
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # White's turn
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],  # White's turn
    ]
    edge_index = [[0, 1, 1, 2], [1, 0, 2, 1]]
    
    # Test conversion to NetworkX
    G = board_to_networkx(node_features, edge_index)
    print(f"Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Test graph hashing
    hash1 = graph_hash(node_features, edge_index)
    print(f"Graph hash: {hash1}")
    
    # Test that identical graphs produce the same hash
    hash2 = graph_hash(node_features, edge_index)
    assert hash1 == hash2, "Same graph should produce same hash"
    print("Hash consistency test passed!")
    
    # Test conversion back to PyG
    node_features_back, edge_index_back = networkx_to_pyg(G)
    print(f"Converted back to PyG format")
    print(f"Node features shape: {len(node_features_back)}x{len(node_features_back[0])}")
    print(f"Edge index shape: {len(edge_index_back)}x{len(edge_index_back[0])}")
    
    print("\nAll tests passed!")
