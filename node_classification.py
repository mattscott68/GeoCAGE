# node_classification.py

import pandas as pd
import numpy as np
import functools as ft
from sklearn import preprocessing, model_selection as sk_ms
from sklearn.svm           import LinearSVC
from sklearn.metrics       import (
    accuracy_score, precision_score,
    recall_score, f1_score, rand_score
)

class NodeClassification:

    def __init__(self, embedding_data, labels, normalize=False):
        if normalize:
            arr = preprocessing.normalize(embedding_data, norm='l2', axis=1)
            self.data = pd.DataFrame(arr)
        else:
            self.data = pd.DataFrame(embedding_data)
        self.labels = labels
        self.oper_math = ['sum', 'avg']
        # configure all your metrics
        self.performal_measure = {
            "accuracy_score":    ft.partial(accuracy_score),

            "precision_macro":   ft.partial(precision_score, average='macro',    zero_division=0),
            "precision_micro":   ft.partial(precision_score, average='micro',    zero_division=0),
            "precision_weighted":ft.partial(precision_score, average='weighted', zero_division=0),

            "recall_macro":      ft.partial(recall_score,    average='macro',    zero_division=0),
            "recall_micro":      ft.partial(recall_score,    average='micro',    zero_division=0),
            "recall_weighted":   ft.partial(recall_score,    average='weighted', zero_division=0),

            "f1_macro":          ft.partial(f1_score,        average='macro',
                                            labels=np.unique(labels),
                                            zero_division=0),
            "f1_micro":          ft.partial(f1_score,        average='micro',
                                            labels=np.unique(labels),
                                            zero_division=0),
            "f1_weighted":       ft.partial(f1_score,        average='weighted',
                                            labels=np.unique(labels),
                                            zero_division=0),
        }

    def split_dataset(self, split_threshold, num_split, random_set):
        splits, _cache = [], None
        for _ in range(num_split):
            if random_set or _cache is None:
                X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(
                    self.data, self.labels, test_size=split_threshold
                )
                _cache = {
                    "X_train": X_train, "X_test": X_test,
                    "Y_train": Y_train, "Y_test": Y_test
                }
            splits.append(_cache)
        return splits

    def classification(self, classifier_name, split_threshold,
                       repetitions, group_by, random_set):
        splits = self.split_dataset(split_threshold, repetitions, random_set)
        if classifier_name != "svm":
            raise NodeClassification_notClassifierFound(classifier_name)
        clf = LinearSVC()
        perf_iters = {}
        for i in range(repetitions):
            data = splits[i]
            clf.fit(data["X_train"], data["Y_train"])
            preds = clf.predict(data["X_test"])
            perf_iters[f"iter_{i}"] = self.performance_computation(data["Y_test"], preds)

        # aggregate
        agg = {op: {m: [] for m in self.performal_measure} for op in group_by}
        for metrics in perf_iters.values():
            for m, v in metrics.items():
                for op in group_by:
                    agg[op][m].append(v)
        for op in agg:
            for m in agg[op]:
                agg[op][m] = (sum(agg[op][m]) / len(agg[op][m])
                              if op == "avg" else sum(agg[op][m]))
        return agg

    def performance_computation(self, Y_true, Y_pred):
        return {
            m: fn(Y_true, Y_pred)
            for m, fn in self.performal_measure.items()
        }

class NodeClassification_notClassifierFound(Exception):
    def __init__(self, name): self.name = name
    def __str__(self):
        return f"No classifier '{self.name}'.  Implemented: 'svm'."

class NodeClassification_notAggregationRecognizer(Exception):
    def __init__(self, name): self.name = name
    def __str__(self):
        return f"Unsupported group_by '{self.name}'. Use 'sum' or 'avg'."
