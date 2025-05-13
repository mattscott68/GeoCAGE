import numpy as np
import nltk.cluster.util as util

# save the original cosine_distance
_orig_cosine = util.cosine_distance

def _safe_cosine(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # if either vector is all zeros, define distance = 1.0
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return 1.0
    return _orig_cosine(x, y)

# patch it in
util.cosine_distance = _safe_cosine


import functools as ft
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, rand_score, davies_bouldin_score, calinski_harabasz_score
from nltk.cluster.util import cosine_distance, euclidean_distance
from nltk.cluster.kmeans import KMeansClusterer


class NodeClustering():

    def __init__(self, data, labels):
        self.data = data
        self.oper_math = ['sum', 'avg']
        self.labels = labels

        self.performal_measure = {
            "rand_score" : ft.partial(rand_score),
            #"davies_bouldin_score" : ft.partial(davies_bouldin_score),
            #"calinski_harabasz_score" : ft.partial(calinski_harabasz_score),
        }

    def classic_clusterizzation(self, repetitions, group_by, base_seed=220):
        """
        repetitions : int, number of repetitions of classification
        group_by   : array, math operation group by measure
        return     : dict of performance measures
        """
        import random
        import numpy as np
        from nltk.cluster.kmeans import KMeansClusterer
        from nltk.cluster.util  import cosine_distance

        # -- 0) pre‑normalize each data‐vector once, zero‐vectors stay zero --
        safe_data = []
        for row in self.data:
            arr = np.asarray(row, dtype=float)
            norm = np.linalg.norm(arr)
            if norm > 0:
                safe_data.append((arr / norm).tolist())
            else:
                safe_data.append(arr.tolist())

        classic_centroid_number = len(np.unique(self.labels))
        measures_performance    = {}

        for i in range(repetitions):
            seed = base_seed + i
            # seed both Python and NumPy
            random.seed(seed)
            np.random.seed(seed)
            # build scikit‑learn KMeans for reproducibility & speed
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=classic_centroid_number,
                random_state=seed
                )
            # safe_data can be a list of lists or a 2D np.ndarray
            Y_pred = kmeans.fit_predict(safe_data)
            perf   = self.performance_computation(self.labels, Y_pred)
            measures_performance[f"iter_{i}"] = perf

        # rest of your aggregation logic unchanged
        measure_total = {}
        for oper in group_by:
            if oper in self.oper_math:
                measure_total[oper] = {m: [] for m in self.performal_measure}
            else:
                raise NodeClustering_notAggregationRecognizer(oper)

        for perf in measures_performance.values():
            for m, val in perf.items():
                for oper in group_by:
                    measure_total[oper][m].append(val)

        for oper in group_by:
            for m in self.performal_measure:
                vals = measure_total[oper][m]
                if oper == "avg":
                    measure_total[oper][m] = sum(vals) / len(vals)
                elif oper == "sum":
                    measure_total[oper][m] = sum(vals)
                else:
                    measure_total[oper][m] = 0

        return measure_total

    def performance_computation(self, Y_test, Y_pred):
        performance_measure_computed = dict()

        for measure_name in self.performal_measure:
            measure_function = self.performal_measure[measure_name]
            measure_value = measure_function(Y_test, Y_pred)
            performance_measure_computed[measure_name] = measure_value

        return performance_measure_computed

class NodeClustering_notAggregationRecognizer(Exception):
      """Exception raised for not classifier type found"""

      def __init__(self, name):
          self.name = name

      def __str__(self):
          return f" Clustering not support '{type(self.name)}' group_by. It's accept: 'sum' or 'avg'."

class NodeClustering_notPerformanceRecognizer(Exception):
      """Exception raised for not classifier type found"""

      def __init__(self, name):
          self.name = name

      def __str__(self):
          return f" Clustering not support '{type(self.name)}' performance measure."