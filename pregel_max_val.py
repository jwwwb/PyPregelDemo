import pregel
import numpy as np
import networkx as nx
import sys


class MaxValAggregator(pregel.Aggregator):
    def __init__(self):
        super(MaxValAggregator, self).__init__()

    def aggregate(self, values):
        return np.vstack(values)


class MaxValCombiner(pregel.Combiner):
    def __init__(self, params=None):
        super(MaxValCombiner, self).__init__()
        self.params = params

    def combine(self, messages):
        tos = []
        values = []
        for message in messages:
            tos.append(message.to)
            values.append(message.value)
        tos = np.asarray(tos)
        values = np.asarray(values)
        tos = tos[np.argsort(-values)]
        values = values[np.argsort(-values)]
        tos = tos[np.argsort(tos)]
        values = values[np.argsort(tos)]
        tos, mask = np.unique(tos, return_index=True)
        values = values[mask]
        new_messages = []
        for m in range(len(values)):
            new_messages.append(pregel.Message(tos[m], values[m]))
        return new_messages


class MaxValVertex(pregel.Vertex):
    def __init__(self, vid, master, edges=None, value=0):
        super(MaxValVertex, self).__init__(vid, master, edges, value)
        self.value = np.random.randint(0, 100)

    def get_aggregation(self):
        return self.vid, self.value

    def compute(self, in_messages):
        # print('Step {}: node {} with value {} receives messages: {}'.format(self.superstep, self.vid, self.value, [m.value for m in in_messages]))
        values = [self.value]
        for message in in_messages:
            values.append(message.value)
        if self.value == max(values) and self.superstep > 0:
            self.vote_to_halt()
            out_messages = []
            # print('Voting to halt.')
        else:
            self.value = max(values)
            out_messages = [pregel.Message(eid, self.value, source=self.vid) for eid in self.edge_ids]
            # print('Sending message with {} to nodes {}'.format(self.value, self.edge_ids))
        return out_messages


def main():
    # set user parameters here
    steps = 8
    vertex_ = getattr(sys.modules[__name__], 'MaxValVertex')
    aggregator_ = getattr(sys.modules[__name__], 'MaxValAggregator')
    combiner_ = getattr(sys.modules[__name__], 'MaxValCombiner')
    # graph = nx.relaxed_caveman_graph(4, 8, 0.2, seed=42)
    # graph = nx.erdos_renyi_graph(10000, 0.15)
    graph = nx.fast_gnp_random_graph(100000, 0.000001)

    # run pregel
    pregel_net = pregel.Master(vertex_, aggregator_, combiner_, graph=graph)
    print('built network')
    listy = np.asarray(pregel_net.aggregate_values())
    print(listy[np.argsort(listy[:, 0])][:, 1])

    pregel_net.run(steps)

    print('\n', pregel_net.superstep, '\n')
    listy = np.asarray(pregel_net.aggregate_values())
    values = listy[np.argsort(listy[:, 0])][:, 1]
    print(values)
    print(values.min(), values.max())


if __name__ == '__main__':
    main()
