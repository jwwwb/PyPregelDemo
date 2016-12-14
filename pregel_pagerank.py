import pregel
import numpy as np
import networkx as nx


class PageRankVertex(pregel.Vertex):
    def __init__(self, vid, master, edges=None, value=0):
        super(PageRankVertex, self).__init__(vid, master, edges, value)

    def compute(self, in_messages):
        if self.superstep == 0:
            self.value = 1.0/self.master.n_nodes
        else:
            valsum = sum([message.value for message in in_messages])
            self.value = 0.15 / self.master.n_nodes + 0.85 * valsum
        out_messages = [pregel.Message(self.value, eid) for eid in self.edge_ids]
        return out_messages
