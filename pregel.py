import numpy as np
import networkx as nx

# n_partitions = 4
# total_nodes = 34


# base classes
class Combiner(object):
    def __init__(self, params=None):
        self.params = params

    def combine(self, messages):
        return messages


class Aggregator(object):
    def __init__(self, params=None):
        self.params = params

    def aggregate(self, values):
        aggregation = None
        return aggregation


class Message(object):
    def __init__(self, to, value=None, source=None):
        self.to = to
        self.value = value
        self.source = source    # only used for the GUI demo

    def __repr__(self):
        return '<Message: {} -> {}>'.format(self.source, self.to)

    def __str__(self):
        return 'Message from {} to {} containing: '.format(self.source, self.to) + str(self.value)


class Edge(object):
    def __init__(self, target, value=1):
        self.target = target
        self.value = value


class Vertex(object):
    def __init__(self, vid, master, edges=None, value=1, params=None):
        self.vid = vid
        self.master = master
        self.params = params
        self.edges = edges if edges else []
        self.value = value
        self.edge_ids = np.zeros(len(self.edges), dtype='uint64')
        self.edge_values = np.zeros(len(self.edges))
        for e in range(len(self.edges)):
            self.edge_ids[e] = self.edges[e].target
            self.edge_values[e] = self.edges[e].value
        self.active = 1
        self.superstep = 0

    def add_edge(self, edge):
        if edge.target not in self.edge_ids:
            self.edges.append(edge)
            self.edge_ids = np.concatenate((self.edge_ids, np.asarray([edge.target], dtype='uint64')))
            self.edge_values = np.concatenate((self.edge_values, np.asarray([edge.value])))

    def vote_to_halt(self):
        self.active = 0

    def compute(self, in_messages):
        out_messages = [Message(0, 'hi!')]
        return out_messages

    def get_aggregation(self):
        return self.value

    def execute_superstep(self, superstep, in_messages):
        assert superstep == self.superstep
        if self.active or len(in_messages) > 0:
            self.active = 1
            out_messages = self.compute(in_messages)
            self.superstep += 1
            return out_messages
        else:
            self.superstep += 1


class Partition(object):
    def __init__(self, pid, master, n_partitions, vertices=None, combiner=Combiner(), aggregator=Aggregator()):
        self.pid = pid
        self.master = master
        self.n_partitions = n_partitions
        self.combiner = combiner
        self.aggregator = aggregator
        self.vertices = vertices if vertices else []
        self.vertex_ids = np.zeros(len(self.vertices))
        for v in range(len(self.vertices)):
            self.vertex_ids[v] = self.vertices[v].id
        self.superstep = 0
        self.in_buffer = []
        self.actives = np.ones(len(self.vertices))

    def add_vertex(self, vertex):
        self.vertices.append(vertex)
        self.vertex_ids = np.concatenate((self.vertex_ids, np.asarray([vertex.vid])))
        self.actives = np.concatenate((self.actives, np.ones(1)))

    def add_edge(self, vertex, edge):
        if vertex in self.vertex_ids:
            self.vertices[np.argwhere(self.vertex_ids == vertex)[0, 0]].add_edge(edge)

    def aggregate_values(self):
        return self.aggregator.aggregate([self.vertices[v].get_aggregation() for v in range(len(self.vertex_ids))])

    def execute_superstep(self, superstep, in_buffer):
        assert superstep == self.superstep
        self.in_buffer.extend(in_buffer)
        out_buffer = []
        vertex_messages = [[] for _ in range(len(self.vertex_ids))]
        for message in self.in_buffer:
            vertex_messages[np.argwhere(self.vertex_ids == message.to)[0, 0]].append(message)
        self.in_buffer = []
        for v in range(len(self.vertices)):
            out_messages = self.vertices[v].execute_superstep(superstep, vertex_messages[v])
            if out_messages:
                for message in out_messages:
                    if message.to in self.vertex_ids:
                        self.in_buffer.append(message)
                    else:
                        out_buffer.append(message)
            self.actives[v] = self.vertices[v].active
        self.superstep += 1
        return self.combiner.combine(out_buffer), self.actives


class Master(object):
    def __init__(self, vertex_type, aggregator_type, combiner_type, vertex_params=None,
                 aggregator_params=None, combiner_params=None,
                 graph=nx.Graph(), n_partitions=3, undirected=True):
        # the loading and division of the graph normally occurs in a distributed fashion also, however for simplicity
        # I will implement it in bulk here.
        aggregator_params = aggregator_params if aggregator_params is not None else {}
        combiner_params = combiner_params if combiner_params is not None else {}
        vertex_params = vertex_params if vertex_params is not None else {}
        self.aggregator = aggregator_type(**aggregator_params)
        self.combiner = combiner_type(**combiner_params)
        self.n_partitions = n_partitions
        self.partitions = []
        self.n_nodes = graph.number_of_nodes()
        for p in range(n_partitions):
            self.partitions.append(Partition(p, self, self.n_partitions, aggregator=aggregator_type(**aggregator_params),
                                             combiner=combiner_type(**combiner_params)))
        self.nodes_active = 0
        for node in graph.nodes_iter():
            self.nodes_active += 1
            self.partitions[self.assignment(node, self.n_partitions)].add_vertex(vertex_type(node, self, **vertex_params))
        for source, target in graph.edges_iter():
            self.partitions[self.assignment(source, self.n_partitions)].add_edge(source, Edge(target))
            if undirected:
                self.partitions[self.assignment(target, self.n_partitions)].add_edge(target, (Edge(source)))
        self.superstep = 0

    def aggregate_values(self):
        return self.aggregator.aggregate([self.partitions[p].aggregate_values() for p in range(self.n_partitions)])

    def execute_superstep(self, messages):
        buffers = [[] for _ in range(self.n_partitions)]
        for message in messages:
            buffers[self.assignment(message.to, self.n_partitions)].append(message)
        messages = []
        actives = np.asarray([])
        for p in range(len(self.partitions)):
            out_buffer, act = self.partitions[p].execute_superstep(self.superstep, buffers[p])
            messages.extend(out_buffer)
            actives = np.concatenate((actives, act))
        self.nodes_active = np.sum(actives)
        self.superstep += 1
        return messages

    def run(self, n_steps):
        messages = []
        print(self.superstep)
        while self.superstep < n_steps and (self.nodes_active > 0 or len(messages) > 0):
            messages = self.execute_superstep(messages)

    @staticmethod
    def assignment(vid, n_partitions):
        return int(vid % n_partitions)


if __name__ == '__main__':
    print('Subclass Vertex() to execute a Pregel instance.')
