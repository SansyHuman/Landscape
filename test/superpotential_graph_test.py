from common.superpotential_parser import *
from common.inconsistents_parser import *
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it


theory = input('Theory name: ')
w = input('Superpotential in list format: ')

w_obj = Superpotential(theory, w)

graph_data = from_networkx(w_obj.superpotential_graph, group_node_attrs=['node_type', 'matter', 'index'])
print('Graph nodes data: ')
print(graph_data.x)
print('Graph edges data: ')
print(graph_data.edge_index)

fig, ax = plt.subplots()
nx.draw(w_obj.superpotential_graph, with_labels=True, font_weight='bold',
        pos=nx.arf_layout(w_obj.superpotential_graph),
        connectionstyle=[f"arc3,rad={r}" for r in it.accumulate([0.03] * len(w_obj.superpotential_graph.edges))],
        ax=ax)

plt.show()