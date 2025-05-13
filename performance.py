import pandas as pd
from sklearn import preprocessing
from visualization import VisualEmbedding
from node_classification import NodeClassification
from node_clustering     import NodeClustering
from graph_e_model_mlp import GraphEModel
import pandas as pd
import matplotlib.pyplot as plt


class PerformanceEmbedding():

    def __init__(self, model, embedding_name='att', node_label='node_label'):
        if not hasattr(model, 'get_embedding'):
            raise PerformanceEmbedding_notModelClass(model)
        self.embedding = model.get_embedding(phase=embedding_name, type_output="numpy")
        self.labels = list(model.get_embedding(phase=node_label, type_output="numpy"))
        self.group_by = ['avg','sum']
        self.cluster_measure = ['rand_score']
        self.classifier_measure = ['accuracy_score','precision_macro','precision_micro','precision_weighted',
                                   'recall_macro','recall_micro','recall_weighted',
                                   'f1_macro','f1_micro','f1_weighted']

    def visualization(self):
        visualemb = VisualEmbedding(self.embedding,self.labels)
        return visualemb.embedding_visualization(None)

    def classification(self, repetitions = 10, classifier_name = "svm", performance_group_by='avg',labeled_data_threshold=None, measures_selected = None, random_set = True):
        if measures_selected is None:
            measures_selected = self.classifier_measure
        else:
            for meas in measures_selected:
                if meas not in self.classifier_measure:
                    raise PerformanceEmbedding_notMeasureExperiment(meas,'NodeClassification')
        if labeled_data_threshold is None:
            labeled_data_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


        n_classify = NodeClassification(self.embedding,self.labels, normalize=True)
        measures = dict()
        for split_threshold in labeled_data_threshold:
            measure = n_classify.classification(classifier_name, split_threshold, repetitions, self.group_by, random_set)
            _key = "split_"+str(split_threshold)
            measures[_key] = measure

        return self.performance_measure(measures,measures_selected,performance_group_by)

    def clusterization(self, repetitions = 10, performance_group_by='avg', measures_selected = None, seeding=220):
        if measures_selected is None:
            measures_selected = self.cluster_measure
        else:
            for meas in measures_selected:
                if meas not in self.cluster_measure:
                    raise PerformanceEmbedding_notMeasureExperiment(meas,'NodeClustering')

        n_clusterfy = NodeClustering(self.embedding,self.labels)
        measures = dict()
        measure = n_clusterfy.classic_clusterizzation(repetitions, self.group_by, base_seed=seeding)
        measures["all"] = measure
        return self.performance_measure(measures,measures_selected,performance_group_by)



    def performance_measure(self, measures, measures_selected, groub_by='avg',):
        pd_measure = pd.DataFrame()
        pd_measure['name_measure'] = measures_selected

        for split_name in measures:
            val_col = list()
            for meas_name in measures_selected:
                if meas_name not in measures[split_name][groub_by]:
                    raise PerformanceEmbedding_notMeasure(meas_name)
                else:
                    val_col.append(measures[split_name][groub_by][meas_name])
            pd_measure[split_name] = val_col
        pd_measure.set_index('name_measure', inplace = True)
        return pd_measure


    def loss_plot(self):
        data_plot_losses = [val.item() for val in DAGE_values['losses']]
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot(data_plot_losses,"b.")



class PerformanceEmbedding_notModelClass(Exception):
      """Exception raised for not classifier type found"""

      def __init__(self, obj):
          self.obj = obj

      def __str__(self):
          return f"Model should be a 'GraphEModel' class object but receive a ''{type(self.obj)} object."

class PerformanceEmbedding_notMeasure(Exception):
      """Exception raised for not classifier type found"""

      def __init__(self, name_measure):
          self.name_measure = name_measure

      def __str__(self):
          return f"Percormance '{self.name_measure}' not recognized."

class PerformanceEmbedding_notMeasureExperiment(Exception):
      """Exception raised for not classifier type found"""

      def __init__(self, measure_name, experiment_name):
          self.measure_name = measure_name
          self.experiment_name = experiment_name

      def __str__(self):
          return f"Experiment '{self.experiment_name}' not implement performance called '{self.measure_name}'."