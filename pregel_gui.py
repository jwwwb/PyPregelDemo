# Main GUI
# 2016-12-06 James W Browne
# CAML seminar visual Pregel demo
#
# This class creates the main GUI for the program, from which all other
# code is run. Implements a QtGui as well as the threading for NetListener
# and to receive final decisions from Classifier. Also calls RecordLabs in
# order to designate labels for a certain timestamp for supervised training.
# Finally can call TrainingSel to potentially select a subset of these
# labels and then train a classifier.


# import classes
from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt

# own classes
import ui_design as uid
import pregel
import pregel_cluster
import pregel_max_val


class MainGUI(QtWidgets.QMainWindow, uid.Ui_MainWindow):
    def __init__(self):
        super(MainGUI, self).__init__()
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        if self.frameGeometry().height() > self.screen.height():
            self.setGeometry(0, 0, self.frameGeometry().width(), self.screen.height())
        self.setupUi(self)
        self.setWindowTitle('Pregel Live Demo')





        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Set User parameters here!!
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """

        self.kind = 'cluster'       # which algorithm to perform / 'cluster' or 'max'
        self.mode = 'linear'        # how to distribute nodes across the workers / 'linear' or 'hash'
        self.max_clusters_global = 6
        self.max_clusters_local = 5
        self.min_similarity = 0.5
        self.boundary_factor = 0.3
        self.max_vertices = 10
        # self.graph = nx.relaxed_caveman_graph(3, 3, 0.25, seed=42)
        # self.graph = nx.relaxed_caveman_graph(3, 4, 0.25, seed=42)
        self.graph = nx.read_adjlist('./seven_graph.txt', nodetype=int)
        # self.reshuffle = [1,0,5,4,3,2]
        # self.reshuffle = [0, 1, 2]
        self.reshuffle = range(self.max_clusters_global)

        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Set User parameters here!!
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """





        self.master = self.make_pregel()
        self.labels = np.zeros(self.graph.number_of_nodes(), dtype='uint32')
        self.best_clusters = np.asarray([range(self.graph.number_of_nodes())])
        self.edges = None
        self.start_of_step = True
        self.messages = []
        self.n_steps = 15
        self.converged = False
        self.step_speed = 20
        self.pushButtonRestart.clicked.connect(self.restart_pregel)
        self.pushButtonStep.clicked.connect(self.execute_step)
        # self.pushButtonRun.clicked.connect(self.run)
        self.pushButtonRun.clicked.connect(self.plot_graph)

        self.pushButtonRun.setText('Plot')
        self.previous_x_positions = [0, 0, 0]
        self.previous_y_positions = [0, 0, 0]

        self.vertex_frames = []
        self.vertex_statuses = []
        self.vertex_numbers = []
        self.vertex_labels = []
        self.vertex_tables = []
        for node in self.graph.nodes_iter():
            self.make_node(node, self.assignment(node))

        self.redraw_nodes([[] for _ in range(self.graph.number_of_nodes())])

    def assignment(self, node):
        if self.mode == 'hash':
            return int(node % 3)+1
        elif self.mode == 'linear':
            return int(node/(self.graph.number_of_nodes()/3.0)+1)

    def make_node(self, nodeno, workno):
        if self.previous_x_positions[workno-1]:
            x, y = self.previous_x_positions[workno-1]+140, (self.previous_y_positions[workno-1]-140)%280
        else:
            x, y = 10, 150
        self.previous_x_positions[workno - 1] = x
        self.previous_y_positions[workno - 1] = y
        _translate = QtCore.QCoreApplication.translate
        worker_frame = eval('self.worker_{}'.format(workno))
        self.vertex_frames.append(QtWidgets.QFrame(worker_frame))
        self.vertex_frames[-1].setGeometry(QtCore.QRect(x, y, 120, 150))
        self.vertex_frames[-1].setAutoFillBackground(True)
        self.vertex_frames[-1].setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.vertex_frames[-1].setFrameShadow(QtWidgets.QFrame.Raised)
        self.vertex_frames[-1].setObjectName("frameVertex_{}".format(nodeno))
        self.vertex_statuses.append(QtWidgets.QPushButton(self.vertex_frames[-1]))
        self.vertex_statuses[-1].setGeometry(QtCore.QRect(0, 0, 40, 40))
        self.vertex_statuses[-1].setObjectName("vertexStatus_{}".format(nodeno))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.vertex_statuses[-1].setFont(font)
        self.vertex_statuses[-1].setText(_translate("MainWindow", " "))
        self.vertex_labels.append(QtWidgets.QLabel(self.vertex_frames[-1]))
        self.vertex_labels[-1].setGeometry(QtCore.QRect(45, 20, 55, 20))
        self.vertex_labels[-1].setObjectName("labelVertex_{}".format(nodeno))
        self.vertex_labels[-1].setText(_translate("MainWindow", "messages"))
        self.vertex_numbers.append(QtWidgets.QLabel(self.vertex_frames[-1]))
        self.vertex_numbers[-1].setGeometry(QtCore.QRect(100, 0, 20, 20))
        self.vertex_numbers[-1].setObjectName("numberVertex_{}".format(nodeno))
        self.vertex_numbers[-1].setFont(font)
        self.vertex_numbers[-1].setText(_translate("MainWindow", "{}".format(nodeno)))
        self.vertex_tables.append(QtWidgets.QTableWidget(self.vertex_frames[-1]))
        self.vertex_tables[-1].setGeometry(QtCore.QRect(0, 40, 120, 110))
        self.vertex_tables[-1].setObjectName("tableWidget_{}".format(nodeno))
        self.vertex_tables[-1].setColumnCount(0)
        self.vertex_tables[-1].setRowCount(0)

    def redraw_nodes(self, msgs):
        self.list_actives.clear()
        self.display_superstep.setText(str(self.master.superstep))
        for wid in self.master.partitions:
            for vert in wid.vertices:
                if self.kind == 'max':
                    messages = msgs[vert.vid]
                    self.vertex_statuses[vert.vid].setText(str(vert.value))
                    if self.start_of_step:
                        self.vertex_labels[vert.vid].setText('in msgs')
                    else:
                        self.vertex_labels[vert.vid].setText('out msgs')
                elif self.kind == 'cluster':
                    self.vertex_labels[vert.vid].setText('clusters')
                    messages = list(zip(['{:.2f}'.format(va) for va in vert.value[0]], vert.value[1]))
                if vert.active:
                    self.vertex_statuses[vert.vid].setStyleSheet('background-color: orange; color: black;')
                    self.list_actives.addItem('{}'.format(vert.vid))
                else:
                    self.vertex_statuses[vert.vid].setStyleSheet('background-color: grey; color: black;')

                self.vertex_tables[vert.vid].clear()
                while self.vertex_tables[vert.vid].columnCount() < 2:
                    self.vertex_tables[vert.vid].insertColumn(self.vertex_tables[vert.vid].columnCount())
                for row in range(len(messages)):
                    if self.vertex_tables[vert.vid].rowCount() <= row:
                        self.vertex_tables[vert.vid].insertRow(row)
                    for i in range(2):
                        self.vertex_tables[vert.vid].setItem(row, i, QtWidgets.QTableWidgetItem(str(messages[row][i])))
                self.vertex_tables[vert.vid].resizeColumnsToContents()
                self.vertex_tables[vert.vid].resizeRowsToContents()
                self.vertex_tables[vert.vid].horizontalHeader().hide()
                self.vertex_tables[vert.vid].verticalHeader().hide()

    def make_pregel(self):
        if self.kind == 'max':
            vertex_ = getattr(pregel_max_val, 'MaxValVertex')
            aggregator_ = getattr(pregel_max_val, 'MaxValAggregator')
            combiner_ = getattr(pregel, 'Combiner')
            master = pregel.Master(vertex_, aggregator_, combiner_, graph=self.graph, n_partitions=3)
            return master
        if self.kind == 'cluster':
            vertex_ = getattr(pregel_cluster, 'SemiClusterVertex')
            aggregator_ = getattr(pregel_cluster, 'SemiClusterAggregator')
            combiner_ = getattr(pregel, 'Combiner')
            aggregator_params = {'similarity': self.min_similarity, 'max_clusters': self.max_clusters_global}
            vertex_params = {'max_vertices': self.max_vertices, 'boundary_factor': self.boundary_factor, 'max_clusters': self.max_clusters_local}
            master = pregel.Master(vertex_, aggregator_, combiner_, aggregator_params=aggregator_params,
                                   vertex_params=vertex_params, graph=self.graph, n_partitions=3)
            return master
        else:
            print('invalid pregel type')

    def get_edge(self, node1, node2):
        u = eval('self.worker_{0}.x()'.format(self.assignment(node1))) + self.vertex_frames[node1].x() + \
            self.vertex_frames[node1].width() / 2 + np.random.randint(-10,10)
        v = eval('self.worker_{0}.y()'.format(self.assignment(node1))) + self.vertex_frames[node1].y() + \
            self.vertex_frames[node1].height() / 2 + np.random.randint(-10,10)
        x = eval('self.worker_{0}.x()'.format(self.assignment(node2))) + self.vertex_frames[node2].x() + \
            self.vertex_frames[node2].width() / 2 + np.random.randint(-10,10)
        y = eval('self.worker_{0}.y()'.format(self.assignment(node2))) + self.vertex_frames[node2].y() + \
            self.vertex_frames[node2].height() / 2 + np.random.randint(-10,10)
        return u, v, x, y

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawLines(qp)
        qp.end()

    def drawLines(self, qp):

        pen = QtGui.QPen(QtCore.Qt.blue, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        for s, d in self.graph.edges_iter():
            u, v, x, y = self.get_edge(s, d)
            qp.drawLine(u, v, x, y)

        # if self.converged:
        #     pen = QtGui.QPen(QtCore.Qt.red, 6, QtCore.Qt.SolidLine)
        #     qp.setPen(pen)
        #     qp.drawLine(20,20,600,600)

    def display_out_messages(self):
        messages = [[] for _ in range(self.graph.number_of_nodes())]
        for msg in self.messages:
            messages[msg.source].append((msg.to, msg.value))
        self.redraw_nodes(messages)

    def display_in_messages(self):
        messages = [[] for _ in range(self.graph.number_of_nodes())]
        for msg in self.messages:
            messages[msg.to].append((' ', msg.value))
        self.redraw_nodes(messages)

    def restart_pregel(self):
        # self.graph = nx.relaxed_caveman_graph(3, 3, 0.2, seed=42)
        self.label_actives.setText('Active Vertices')
        self.master = self.make_pregel()
        self.edges = None
        self.start_of_step = True
        self.messages = []
        self.n_steps = 15
        self.converged = False
        self.redraw_nodes([[] for _ in range(self.graph.number_of_nodes())])

    def plot_graph(self):
        if self.converged:
            labels = np.zeros([self.graph.number_of_nodes(), 12], dtype='uint32')
            for c, cluster in enumerate(self.best_clusters):
                labels[cluster, c] += 1
            print(labels)
            labels = labels.transpose()
            labels = labels[self.reshuffle]
            # labels = labels[[0, 1, 2]]
            for l in range(len(labels)):
                labels[l] *= (len(labels)-l)
            labels = labels.transpose()
            print(labels)
            for n, node in enumerate(labels):
                if np.sum(node) == 0:
                    self.labels[n] = 0
                else:
                    self.labels[n] = np.mean(node[node>0])
            nx.draw_spring(self.graph, with_labels=True, node_color=self.labels)
            plt.show()

    def run(self):
        self.step_speed = self.spinBoxSpeed.value()
        while not self.converged:
            self.execute_step()
            time.sleep(0.1 * self.step_speed)

    def execute_step(self):
        if self.kind == 'cluster':
            if not self.converged:
                if self.master.superstep < self.n_steps and (self.master.nodes_active > 0 or len(self.messages) > 0):
                    self.messages = self.master.execute_superstep(self.messages)
                    self.redraw_nodes('this shouldnt do anything')
                else:
                    self.converged = True
                    print('algorithm converged!')
            else:
                scores, best_clusters, _ = self.master.aggregate_values()
                self.label_actives.setText('Best Clusters')
                self.list_actives.clear()
                for s in range(len(scores)):
                    self.list_actives.addItem('{:.2f}:   '.format(scores[s])+str(best_clusters[s]))
                self.best_clusters = best_clusters
                # self.repaint()
        elif self.kind == 'max':
            if not self.converged:
                if self.start_of_step:
                    self.display_in_messages()
                    self.start_of_step = not self.start_of_step
                else:
                    if self.master.superstep < self.n_steps and (self.master.nodes_active > 0 or len(self.messages) > 0):
                        self.messages = self.master.execute_superstep(self.messages)
                        self.display_out_messages()
                        self.start_of_step = not self.start_of_step
                    else:
                        self.converged = True
                        print('algorithm converged!')

def main():
    app = QtWidgets.QApplication(sys.argv)
    form = MainGUI()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
