import json
import networkx as nx
import numpy as np
from collections import OrderedDict

class LoadDataset():

    def __init__(self,edge_file_name,attribute_file_name,label_file_name,attribute_file_format="normalized_matrix",is_directed_graph=False):
        """
        edge_file : file with all edge (pair of nodes)
        attribute_file : file with all attribute
        label_file :
        attribute_file_format : format od attribute data:
              "normal_matrix" : each row is alredy a frequency normalizzated vector (DEFAULT) es: CORA dataset
              "naive_text" : each row is item text description
        is_directed_graph : boolean, if true is a direct graph else (DEFAULT) is a undirect graph

        """
        self.is_directed_graph = is_directed_graph
        #input shape
        self.input_shape = dict()

        #Structural preprocessing
        self.edge_file_name = edge_file_name
        self.graph = self.edge_createGraph()
        self.edge_adj_matrix = np.array(nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes())))

        #Attribute preprocessing
        self.attribute_file_name = attribute_file_name
        self.attribute_adj_matrix = np.array(self.attribute_createMatrix(attribute_file_format))

        #Class preprocessing
        self.label_file_name = label_file_name
        self.label_vec = self.labels_createVector()




    def get_structural_matrix(self):
        return self.edge_adj_matrix

    def get_attribute_matrix(self):
        return self.attribute_adj_matrix

    def get_vector_matrix(self):
        return self.label_vec

    def get_graph(self):
        return self.graph

    def export_graph(self, pathfile, filename, extention="graphml"):
        path = pathfile+'/'+filename+'.'+extention

        if extention == "graphml":
            nx.write_graphml( self.graph, path)
        elif extention == "gml":
            nx.write_gml( self.graph, path)
        else:
            raise LoadDataset_Exception_Graph_FormatExport_notRecognized(extention)
        return True

    def get_input_shape(self, key):
        return self.input_shape[key]

    def edge_createGraph(self):
        if self.is_directed_graph:
            g = nx.DiGraph()
        else:
            g = nx.Graph()
        try:
            with open(self.edge_file_name, 'r') as edge_file:
                for line in edge_file:
                    edge = line.split()
                    if len(edge) == 3:
                        edge_weight = float(edge[2])
                    else:
                        edge_weight = 1.0
                    if len(edge) == 1:
                        g.add_node(int(edge[0]))
                    else:
                        g.add_edge(int(edge[0]), int(edge[1]), weight = edge_weight)
        except Exception as e:
            raise e
        self.input_shape['net'] = g.number_of_nodes()
        print("Structure dimension:\t",self.input_shape['net'])
        return g

    def attribute_createMatrix(self, attribute_file_format):
        if attribute_file_format == "normalized_matrix":
            try:
                att_matrix = []
                with open(self.attribute_file_name, 'r') as att_file:
                    for line in att_file:
                      att_line = line.replace("\n", "").split(" ")[1:]
                      att_matrix.append([float(n) for n in att_line])
                self.input_shape['att'] = len(att_matrix[0])
                print("Attribute dimension:\t",self.input_shape['att'])
                return att_matrix
            except Exception as e:
                raise e
        elif attribute_file_format == "naive_text":
            print("naive_text to do")
            try:
                att_matrix = []
                with open(self.attribute_file_name, 'r') as att_file:
                    for line in att_file:
                        print(line)
                        break
                    corpus = json.load(att_file)
                    print(corpus)

                return 0
            except Exception as e:
                raise e
        else:
            raise(LoadDataset_Exception_Attribute_Format_notRecognized(attribute_file_format))


    def labels_createVector(self):
        try:
            with open(self.label_file_name, 'r') as label_file:
                node_label_dict = {}
                for line in label_file:
                    split_line = line.replace("\n", "").split(" ")
                    node_id = int(split_line[0])
                    node_label = int(split_line[1])
                    node_label_dict[node_id] = node_label
                # sort the keys (node_ids) of the dictionary
                node_label_dict = OrderedDict(sorted(node_label_dict.items(), key=lambda t: t[0]))
                labels = np.array(list(node_label_dict.values()))
                return labels
        except Exception as e:
            raise e

    def get_labels(self):
        return self.label_vec


class LoadDataset_Exception_Attribute_Format_notRecognized(Exception):
      """Exception raised for errors in list of layers type"""

      def __init__(self, value):
          self.value = value

      def __str__(self):
          return f'{self.value} : type of attribute file format not recognized.'

class LoadDataset_Exception_Graph_FormatExport_notRecognized(Exception):
      """Exception raised for errors in list of layers type"""

      def __init__(self, value):
          self.value = value

      def __str__(self):
          return f'{self.value} : graph format export not recognized.'