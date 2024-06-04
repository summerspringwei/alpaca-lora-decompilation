import matplotlib.pyplot as plt
import networkx as nx
import pydot
path = "validation_decompilation/result_bbcount-2_beamsearch-16_4.1K_AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat_head-100/AnghaBench-ll-O2/beanstalkd/extr_file.c_fileincref/.fileincref.dot"
graphs = pydot.graph_from_dot_file(path)
graph = graphs[0]
print(graph)
for node in graph.get_nodes():
    print(node.obj_dict["attributes"]["label"].split("\l"))
    print(node.obj_dict["parent_node_list"])
for edge in graph.get_edge_list():
    print(edge.get_source(), edge.get_destination())
# g = nx.nx_pydot.read_dot(path)
# print(g)
# nx.draw(g, with_labels=True)
# plt.savefig("filename.png")

