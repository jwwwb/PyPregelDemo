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
        self.graph = nx.relaxed_caveman_graph(3, 3, 0.25, seed=21)
        self.master = self.make_pregel('cluster')
        self.edges = None
        self.start_of_step = True
        self.messages = []
        self.n_steps = 8
        self.converged = False
        self.step_speed = 20
        self.shuffler = [1, 6, 7, 2, 5, 8, 3, 4, 9]
        self.pushButtonRestart.clicked.connect(self.restart_pregel)
        self.pushButtonStep.clicked.connect(self.execute_step)
        self.pushButtonRun.clicked.connect(self.run)
        self.redraw_nodes([[] for i in range(self.graph.number_of_nodes())])

        self.vertex_frames = []
        self.vertex_statuses = []
        self.vertex_numbers = []
        self.vertex_labels = []
        self.vertex_tables = []
        self.make_node(0,2)

    def make_node(self, nodeno, workno):
        _translate = QtCore.QCoreApplication.translate
        worker_frame = eval('self.worker_{}'.format(workno))
        self.vertex_frames.append(QtWidgets.QFrame(worker_frame))
        self.vertex_frames[-1].setGeometry(QtCore.QRect(10, 170, 100, 140))
        self.vertex_frames[-1].setAutoFillBackground(True)
        self.vertex_frames[-1].setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.vertex_frames[-1].setFrameShadow(QtWidgets.QFrame.Raised)
        self.vertex_frames[-1].setObjectName("frameVertex_{}".format(nodeno))
        self.vertex_statuses.append(QtWidgets.QPushButton(self.vertex_frames[-1]))
        self.vertex_statuses[-1].setGeometry(QtCore.QRect(0, 0, 40, 40))
        self.vertex_statuses[-1].setObjectName("vertexStatus_{}".format(nodeno))
        self.vertex_statuses[-1].setText(_translate("MainWindow", "v2"))
        self.vertex_labels.append(QtWidgets.QLabel(self.vertex_frames[-1]))
        self.vertex_labels[-1].setGeometry(QtCore.QRect(40, 20, 40, 20))
        self.vertex_labels[-1].setObjectName("labelVertex_{}".format(nodeno))
        self.vertex_labels[-1].setText(_translate("MainWindow", "msgs"))
        self.vertex_numbers.append(QtWidgets.QLabel(self.vertex_frames[-1]))
        self.vertex_numbers[-1].setGeometry(QtCore.QRect(80, 0, 20, 20))
        self.vertex_numbers[-1].setObjectName("numberVertex_{}".format(nodeno))
        font = QtGui.QFont().setPointSize(20)
        self.vertex_numbers[-1].setFont(font)
        self.vertex_numbers[-1].setText(_translate("MainWindow", "{}".format(nodeno)))
        self.vertex_tables.append(QtWidgets.QTableWidget(self.vertex_frames[-1]))
        self.vertex_tables[-1].setGeometry(QtCore.QRect(0, 40, 100, 100))
        self.vertex_tables[-1].setObjectName("tableWidget_{}".format(nodeno))
        self.vertex_tables[-1].setColumnCount(0)
        self.vertex_tables[-1].setRowCount(0)

    def redraw_nodes(self, msgs):
        self.list_actives.clear()
        self.display_superstep.setText(str(self.master.superstep))
        for wid in self.master.partitions:
            for vert in wid.vertices:
                messages = msgs[vert.vid]
                status = eval('self.vertexStatus__{}'.format(self.shuffler[vert.vid]))
                frame = eval('self.frameVertex__{}'.format(self.shuffler[vert.vid]))
                table = eval('self.tableWidget_{}'.format(self.shuffler[vert.vid]))
                label = eval('self.labelVertex__{}'.format(self.shuffler[vert.vid]))
                if vert.active:
                    status.setStyleSheet('background-color: orange; color: black;')
                    self.list_actives.addItem('{}'.format(vert.vid))
                else:
                    status.setStyleSheet('background-color: grey; color: black;')
                status.setText(str(vert.value))
                table.clear()
                while table.columnCount() < 2:
                    table.insertColumn(table.columnCount())
                for row in range(len(messages)):
                    if table.rowCount() <= row:
                        table.insertRow(row)
                    for i in range(2):
                        table.setItem(row, i, QtWidgets.QTableWidgetItem(str(messages[row][i])))
                table.resizeColumnsToContents()
                table.resizeRowsToContents()
                table.horizontalHeader().hide()
                table.verticalHeader().hide()

    def make_pregel(self, kind='max'):
        if kind == 'max':
            vertex_ = getattr(pregel_max_val, 'MaxValVertex')
            aggregator_ = getattr(pregel_max_val, 'MaxValAggregator')
            combiner_ = getattr(pregel, 'Combiner')
            master = pregel.Master(vertex_, aggregator_, combiner_, graph=self.graph, n_partitions=3)
            return master
        if kind == 'cluster':
            vertex_ = getattr(pregel_cluster, 'SemiClusterVertex')
            aggregator_ = getattr(pregel_cluster, 'SemiClusterAggregator')
            combiner_ = getattr(pregel, 'Combiner')
            aggregator_params = {'similarity': 0.6, 'max_clusters': 8}
            vertex_params = {'max_vertices': 10, 'boundary_factor': 0.3, 'max_clusters': 4}
            master = pregel.Master(vertex_, aggregator_, combiner_, aggregator_params=aggregator_params,
                                   vertex_params=vertex_params, graph=self.graph, n_partitions=3)
            return master
        else:
            print('invalid pregel type')

    def get_connecting_line(self, node1, node2):
        u = eval('self.worker_{0}.x()+self.frameVertex__{1}.x()+self.frameVertex__{1}.width()/2'.format((node1 + 2) // 3, node1))
        v = eval('self.worker_{0}.y()+self.frameVertex__{1}.y()+self.frameVertex__{1}.height()/2'.format((node1 + 2) // 3, node1))
        x = eval('self.worker_{0}.x()+self.frameVertex__{1}.x()+self.frameVertex__{1}.width()/2'.format((node2 + 2) // 3, node2))
        y = eval('self.worker_{0}.y()+self.frameVertex__{1}.y()+self.frameVertex__{1}.height()/2'.format((node2 + 2) // 3, node2))
        return u, v, x, y

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawLines(qp)
        qp.end()

    def drawLines(self, qp):

        pen = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine)

        qp.setPen(pen)
        for s, d in self.graph.edges_iter():
            u, v, x, y = self.get_connecting_line(self.shuffler[s], self.shuffler[d])
            qp.drawLine(u, v, x, y)

    def display_out_messages(self):
        messages = [[] for i in range(self.graph.number_of_nodes())]
        for msg in self.messages:
            messages[msg.source].append((msg.to, msg.value))
        self.redraw_nodes(messages)

    def display_in_messages(self):
        messages = [[] for i in range(self.graph.number_of_nodes())]
        for msg in self.messages:
            messages[msg.to].append((' ', msg.value))
        self.redraw_nodes(messages)

    def restart_pregel(self):
        self.graph = nx.relaxed_caveman_graph(3, 3, 0.2, seed=42)
        self.master = self.make_pregel('max')
        self.edges = None
        self.start_of_step = True
        self.messages = []
        self.n_steps = 8
        self.converged = False

    def run(self):
        self.step_speed = self.spinBoxSpeed.value()
        while not self.converged:
            self.execute_step()
            time.sleep(0.1 * self.step_speed)

    def execute_step(self):
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
