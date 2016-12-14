import pregel
import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt


class SemiClusterAggregator(pregel.Aggregator):
    def __init__(self, similarity=1, max_clusters=np.inf):
        super(SemiClusterAggregator, self).__init__()
        self.similarity = similarity
        self.new_clusters = []
        self.max_clusters = max_clusters

    def find(self, entry):
        if self.similarity == 1:
            for row in self.new_clusters:
                if np.array_equal(entry, row):
                    return True
        else:
            for row in self.new_clusters:
                if self.similar(entry, row) > self.similarity:
                    return True
        return False

    @staticmethod
    def similar(v1, v2):
        return float(len(np.intersect1d(v1, v2, assume_unique=True))) / len(np.union1d(v1, v2))

    def aggregate(self, values):
        scores = np.asarray([])
        clusters = []
        factors = np.zeros((0, 3))
        for value in values:
            scores = np.concatenate((scores, value[0]))
            clusters.extend(value[1])
            factors = np.concatenate((factors, value[2]))
        clusters = np.asarray(clusters)
        top_clusters = np.argsort(scores)
        self.new_clusters = []
        new_scores = []
        new_factors = []
        for cl in range(len(scores) - 1, -1, -1):
            if len(new_scores) < self.max_clusters and not self.find(clusters[top_clusters][cl]):
                self.new_clusters.append(clusters[top_clusters][cl])
                new_scores.append(scores[top_clusters][cl])
                new_factors.append(factors[top_clusters][cl])
        new_value = (np.asarray(new_scores), np.asarray(self.new_clusters), np.asarray(new_factors))
        return new_value


class SemiClusterVertex(pregel.Vertex):
    def __init__(self, vid, master, edges=None, value=0, max_vertices=10, boundary_factor=0.3, max_clusters=10):
        super(SemiClusterVertex, self).__init__(vid, master, edges, value)
        self.aggregator = SemiClusterAggregator(1, max_clusters)
        self.max_vertices = max_vertices
        self.boundary_factor = boundary_factor
        self.semi_clusters = [np.asarray([self.vid])]
        self.scores = np.asarray([0])
        self.factors = np.asarray([[0, np.sum(self.edge_values), 1]])
        self.value = (self.scores, self.semi_clusters, self.factors)

    def compute(self, in_messages):
        if self.superstep == 0:
            self.factors = np.asarray([[0, np.sum(self.edge_values), 1]])
            out_messages = [pregel.Message(eid, (self.scores, self.semi_clusters, self.factors), source=self.vid) for eid in self.edge_ids]
        else:
            previous_score_sum = np.sum(self.value[0])
            previous_candidates = len(self.value[0])
            new_scores = []
            new_clusters = []
            new_factors = []
            for o in range(len(self.value[0])):
                new_clusters.append(np.copy(self.value[1][o]))
                new_scores.append(self.value[0][o])
                new_factors.append(np.copy(self.value[2][o]))
            for message in in_messages:
                msg = message.value
                for m in range(len(msg[0])):
                    if self.vid not in msg[1][m] and len(msg[1][m]) < self.max_vertices:
                        new_clusters.append(np.sort(np.concatenate((msg[1][m], np.asarray([self.vid])))))
                        I = msg[2][m, 0]
                        B = msg[2][m, 1]
                        V = msg[2][m, 2] + 1
                        for e in range(len(self.edge_ids)):
                            if self.edge_ids[e] in msg[1][m]:
                                I += self.edge_values[e]
                                B -= self.edge_values[e]
                            else:
                                B += self.edge_values[e]
                        new_scores.append(2*(I-self.boundary_factor*B)/(V*(V-1)))
                        new_factors.append(np.asarray([I, B, V]))
                    else:
                        new_clusters.append(np.copy(msg[1][m]))
                        new_scores.append(msg[0][m])
                        new_factors.append(np.copy(msg[2][m]))
            temp = (np.asarray(new_scores), np.asarray(new_clusters), np.asarray(new_factors))
            self.value = self.aggregator.aggregate([temp])
            new_score_sum = np.sum(self.value[0])
            new_candidates = len(self.value[0])
            if new_score_sum > previous_score_sum or new_candidates > previous_candidates:
                out_messages = [pregel.Message(eid, self.value, source=self.vid) for eid in self.edge_ids]
            else:
                self.vote_to_halt()
                out_messages = []
        return out_messages


def main():
    # set user parameters here
    steps = 8
    vertex_ = getattr(sys.modules[__name__], 'SemiClusterVertex')
    aggregator_ = getattr(sys.modules[__name__], 'SemiClusterAggregator')
    combiner_ = getattr(pregel, 'Combiner')
    aggregator_params = {'similarity': 0.8, 'max_clusters': 10}
    vertex_params = {'max_vertices': 10, 'boundary_factor': 0.3, 'max_clusters': 10}
    graph = nx.relaxed_caveman_graph(4, 8, 0.2, seed=42)

    # run pregel
    pregel_net = pregel.Master(vertex_, aggregator_, combiner_, aggregator_params=aggregator_params, vertex_params=vertex_params, graph=graph)
    pregel_net.run(steps)
    scores, best_clusters, _ = pregel_net.aggregate_values()
    communities = np.zeros(graph.number_of_nodes())
    for c, cluster in enumerate(best_clusters):
        communities[cluster] += (2**c)*np.ones(cluster.shape)

    # show result
    print('\n\nfinal result\n')
    print('scores:', scores)
    print(best_clusters)
    print(communities)
    nx.draw_spring(graph, with_labels=True, node_color=communities)
    plt.show()


if __name__ == '__main__':
    main()
