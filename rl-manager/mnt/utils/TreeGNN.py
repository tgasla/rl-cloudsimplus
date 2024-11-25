import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class TreeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TreeGNN, self).__init__()
        # GNN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Fully connected layer for the global embedding
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # Global pooling to create a single embedding
        x = global_mean_pool(x, batch)
        # Final embedding
        x = self.fc(x)
        return x


def create_large_tree(num_nodes):
    """
    Create a tree graph with `num_nodes` nodes.
    Returns the graph data object compatible with PyTorch Geometric.
    """
    # Random integer features for each node
    node_features = torch.randint(
        1, 101, (num_nodes, 1), dtype=torch.float
    )  # Values between 1 and 100

    # Generate a tree structure (parent-child relationships)
    edge_index = []
    for i in range(1, num_nodes):
        parent = (i - 1) // 2  # Assuming a binary tree structure
        edge_index.append([parent, i])  # Parent to child
        edge_index.append([i, parent])  # Child to parent (for undirected graph)

    edge_index = (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    )  # Convert to PyG format

    # Single graph, so batch index is all zeros
    batch = torch.zeros(num_nodes, dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, batch=batch)


# Define the model
input_dim = 1  # Dimension of node features
hidden_dim = 64  # Hidden dimension in the GNN layers
output_dim = 16  # Desired embedding size
model = TreeGNN(input_dim, hidden_dim, output_dim)

# Create a tree graph with 161 nodes
num_nodes = 161
tree_graph = create_large_tree(num_nodes)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Forward pass
model.train()
embedding = model(tree_graph.x, tree_graph.edge_index, tree_graph.batch)

print("Tree Embedding:", embedding)
print("Embedding Shape:", embedding.shape)
