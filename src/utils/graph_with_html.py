import networkx as nx
from pyvis.network import Network
import random
import os
# Load the GraphML file
G = nx.read_graphml("../graphrag/examples/amazon/graph_chunk_entity_relation.graphml")

# Create a Pyvis network
net = Network(height="100vh", notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)


# Add colors and title to nodes
for node in net.nodes:
    node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    if "description" in node:
        node["title"] = node["description"]

# Add title to edges
for edge in net.edges:
    if "description" in edge:
        edge["title"] = edge["description"]

# Save and display the network
net.show("knowledge_graph.html")

# Define the output folder
output_folder = "./output"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the network HTML file in the output folder
output_path = os.path.join(output_folder, "knowledge_graph.html")
net.save_graph(output_path)

print(f"Knowledge graph saved to: {output_path}")

