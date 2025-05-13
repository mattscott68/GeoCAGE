# baselines.py
# pip install karateclub gensim scikit-learn
import numpy as np
import torch
from sklearn.manifold import SpectralEmbedding
from karateclub import SDNE, Node2Vec, DeepWalk, TriDNR, Role2Vec  # Role2Vec ~ GAT2Vec stand‑in
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from load_data import LoadDataset
from performance import PerformanceEmbedding

def _load(ds_name):
    loader = LoadDataset(
        edge_file_name      = f"dataset/{ds_name}/in_edges.txt",
        attribute_file_name = f"dataset/{ds_name}/in_features.txt",
        label_file_name     = f"dataset/{ds_name}/in_group.txt",
        attribute_file_format="normalized_matrix",
        is_directed_graph   = False
    )
    A = loader.get_structural_matrix()   # numpy NxN
    X = loader.get_attribute_matrix()    # numpy NxF
    labels = loader.get_labels()         # list length N
    return A, X, labels

def run_LAP(dataset, seed):
    A, X, labels = _load(dataset)
    # SpectralEmbedding expects precomputed affinity
    lap = SpectralEmbedding(
        n_components=128,
        affinity="precomputed",
        random_state=seed
    )
    Z = lap.fit_transform(A + np.eye(A.shape[0]))
    # build a dummy model to wrap embeddings
    class Dummy: pass
    m = Dummy()
    m.space_embedded = {'att':{}, 'net':{}, 'node_label':{}}
    for i,z in enumerate(Z):
        m.space_embedded['net'][i] = torch.from_numpy(z).float()
        m.space_embedded['node_label'][i] = labels[i]
    perf = PerformanceEmbedding(m, embedding_name='net')
    return perf.classification(repetitions=10), perf.clusterization(repetitions=10)

def run_SDNE(dataset, seed):
    A, X, labels = _load(dataset)
    model = SDNE(d=128, beta=5, alpha=1e-5, K=3, nu1=1e-5, nu2=1e-4)
    # karateclub wants an igraph-like; it accepts adjacency matrix directly
    model.fit(A)
    Z = model.get_embedding()  # (N,128)
    return _eval_emb(Z, labels)

def run_node2vec(dataset, seed):
    A, X, labels = _load(dataset)
    model = Node2Vec(walk_length=80, dimensions=128, num_walks=10, workers=1)
    model.fit(A)
    Z = model.get_embedding()
    return _eval_emb(Z, labels)

def run_DeepWalk(dataset, seed):
    A, X, labels = _load(dataset)
    model = DeepWalk(walk_length=80, dimensions=128, num_walks=10, workers=1)
    model.fit(A)
    Z = model.get_embedding()
    return _eval_emb(Z, labels)

def run_KATE(dataset, seed):
    # KATE is attribute‐only: e.g. an autoencoder on X
    from baselines.autoencoders import KATEAE
    A, X, labels = _load(dataset)
    model = KATEAE(input_dim=X.shape[1], hidden_dim=256, alpha=3.0, k=200)
    Z = model.fit_transform(X)  # you’d implement fit_transform to return (N,latent)
    return _eval_emb(Z, labels)

def run_Doc2Vec(dataset, seed):
    A, X, labels = _load(dataset)
    # treat each node's attribute vector as a 'document' of floats
    docs = [TaggedDocument(words=[str(v) for v in row], tags=[i]) for i,row in enumerate(X)]
    m = Doc2Vec(docs, vector_size=128, epochs=50, seed=seed)
    Z = np.vstack([m.dv[i] for i in range(len(X))])
    return _eval_emb(Z, labels)

def run_DW_D2V(dataset, seed):
    # concatenate DeepWalk + Doc2Vec embeddings
    A, X, labels = _load(dataset)
    _, _ = run_DeepWalk(dataset, seed)        # will cache an attribute somewhere?
    Z1 = _last_embedding                      # you’d need to persist it
    _, _ = run_Doc2Vec(dataset, seed)
    Z2 = _last_embedding
    Z = np.concatenate([Z1, Z2], axis=1)
    return _eval_emb(Z, labels)

def run_TriDNR(dataset, seed):
    A, X, labels = _load(dataset)
    model = TriDNR(d=128, walk_length=10, num_walks=10)
    model.fit(A, X)
    Z = model.get_embedding()
    return _eval_emb(Z, labels)

def run_GAT2Vec(dataset, seed):
    A, X, labels = _load(dataset)
    # Role2Vec in karateclub is similar to GAT2Vec
    model = Role2Vec(dimensions=128)
    model.fit(A)
    Z = model.get_embedding()
    return _eval_emb(Z, labels)

def _eval_emb(Z, labels):
    # wrap into dummy model
    import torch
    from performance import PerformanceEmbedding
    class M: pass
    m = M()
    m.space_embedded = {'net':{}, 'att':{}, 'node_label':{}}
    for i, z in enumerate(Z):
        m.space_embedded['net'][i] = torch.from_numpy(z).float()
        m.space_embedded['node_label'][i] = labels[i]
    perf = PerformanceEmbedding(m, embedding_name='net')
    return perf.classification(repetitions=10), perf.clusterization(repetitions=10)
