#!/usr/bin/env python3
import os
import random
import numpy as np
import torch
# ── reproducibility ─────────────────────────────────────────────────────
random.seed(220)
np.random.seed(220)
torch.manual_seed(220)
torch.cuda.manual_seed_all(220)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

import torch
import numpy as np
from utils        import Util_class
from load_data    import LoadDataset
from batch_generator import DataBatchGenerator
from graph_e_model_mlp import GraphEModel
from performance  import PerformanceEmbedding

if __name__ == "__main__":
    # 0) Prepare directories
    os.makedirs("models_checkpoint", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    # 1) Copy your Cora files into ./dataset/cora
    #    (Do this once, outside of Python, or use Python's shutil.copy.)
    #    For example:
    # import shutil
    # shutil.copytree("CAGE/data/cora", "dataset/cora", dirs_exist_ok=True)

    dataset_name = "cora"
    data_dir = os.path.join("CAGE", "data", dataset_name)

    edge_file     = os.path.join(data_dir, "in_edges.txt")
    feature_file  = os.path.join(data_dir, "in_features.txt")
    label_file    = os.path.join(data_dir, "in_group.txt")

    # 2) Load
    loader = LoadDataset(
        edge_file_name=edge_file,
        attribute_file_name=feature_file,
        label_file_name=label_file,
        attribute_file_format="normalized_matrix",
        is_directed_graph=False
    )

    net_adj   = loader.get_structural_matrix()
    att_mat   = loader.get_attribute_matrix()
    labels    = loader.get_labels()

    # 3) Batch generator
    batch_size           = 64
    shuffle              = True
    net_hadamard_coeff   = 5.0
    att_hadamard_coeff   = 5.0
    batchGenerator = DataBatchGenerator(
        net_adj, att_mat, labels,
        batch_size, shuffle,
        net_hadamard_coeff, att_hadamard_coeff
    )

    # 4) Model configuration
    cfg = {
        "net_dim": loader.get_input_shape("net"),
        "net_layers_list": [
            {"type": "DENSE", "features": 128, "act_funtion": "RELU", "bias": True}
        ],
        "net_latent_dim": 128,
        "att_dim": loader.get_input_shape("att"),
        "att_layers_list": [
            {"type": "KCOMP", "ktop": 200, "alpha_factor": 3.0},
            {"type": "DENSE", "features": 128, "act_funtion": "RELU", "bias": True}
        ],
        "att_latent_dim": 128,
        "loss_functions": {
            "net": [
                {"loss_name": "structur_proximity_1order", "coef": 1},
                {"loss_name": "structur_proximity_2order", "coef": 1}
            ],
            "att": [
                {"loss_name": "semantic_proximity_2order", "coef": 1},
                {"loss_name": "square_diff_embedding_proximity", "coef": 1}
            ]
        },
        "optimizator_net": {"opt_name": "adam", "lr_rate": 1e-4},
        "optimizator_att": {"opt_name": "adam", "lr_rate": 1e-3},
        "regularization_net": [{"reg_name": "L2", "coeff": 0.001}],
        "regularization_att": [{"reg_name": "L2", "coeff": 0.001}],
        "checkpoint_config": {
            "type": ["best_model_loss", "first_train", "last_train"],
            "times": 20,
            "overwrite": False,
            "path_file": "models_checkpoint",
            "name_file": f"CAGE_{dataset_name}_checkpoint",
            "path_not_exist": "create",
            "path_exist": "use"
        },
        "model_name": f"CAGE_{dataset_name}_mlp",
        "training_config": "N>A"
    }

    # 5) Instantiate & train
    model = GraphEModel(cfg)
    results_mlp = model.models_training(
        datagenerator = batchGenerator,
        epochs        = {"net": 10, "att": 10},
        path_embedding= "embeddings/"
    )

    # 6) Evaluate
    # … your existing classification code …
    CAGE     = PerformanceEmbedding(model, embedding_name='att')
    df_att = CAGE.classification()
    print("=== ATT Classification (accuracy, etc.) ===")
    print(df_att)

    # now also compute rand score via clustering
    df_clust = CAGE.clusterization(
        repetitions         = 10,
        performance_group_by="avg"
    )
    print("=== ATT Clustering (Rand Score) ===")
    # Try both possibilities: name_measure as column or as index
    if 'name_measure' in df_clust.columns:
        # name_measure is a column
        rand_val = df_clust.loc[df_clust['name_measure']=='rand_score', 'all'].iloc[0]
    else:
        # name_measure is the index
        rand_val = df_clust.loc['rand_score', 'all']
    print(rand_val)
