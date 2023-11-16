import matplotlib.pyplot as plt
import networkx as nx

def draw_neural_network(layer_sizes):
    G = nx.DiGraph()
    pos = {}

    v_spacing = 1
    h_spacing = 1.0

    # Create nodes
    for i, layer_size in enumerate(layer_sizes):
        layer_name = f'Layer {i + 1}'
        for j in range(layer_size):
            node_name = f'{i + 1}-{j + 1}'
            G.add_node(node_name)
            pos[node_name] = (i * h_spacing, v_spacing * (layer_size - 1) / 2.0 - j * v_spacing)

         # Label the layers with increased height
        plt.text(i * h_spacing, v_spacing * (layer_size - 1) / 2.0 + 2.5, layer_name, ha='center', va='center', fontsize=10, color='blue')

    # Create edges
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                G.add_edge(f'{i + 1}-{j + 1}', f'{i + 2}-{k + 1}')

    # Draw the graph with larger black nodes
    nx.draw(G, pos, with_labels=False, node_size=50, node_color='black', font_size=8, font_color='white', font_weight='bold', arrowsize=20)

    # Add circles for each node
    for node, (x, y) in pos.items():
        plt.Circle((x, y), 0.1, color='w', ec='k', zorder=10)

    plt.axis('off')
    plt.show()

# Define the architecture of the neural network
layer_sizes = [20, 3, 3, 1]  # Input layer, 2 hidden layers, and output layer

# Draw the neural network
draw_neural_network(layer_sizes)
