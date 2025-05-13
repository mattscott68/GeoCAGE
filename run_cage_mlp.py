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
import random
import argparse

import numpy as np
import torch

from utils              import Util_class
from load_data          import LoadDataset
from batch_generator    import DataBatchGenerator
from graph_e_model_mlp  import GraphEModel
from performance        import PerformanceEmbedding

def run_mlp(dataset_name: str, seed: int = 0):
    # ── 0) fix random seeds ─────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── 1) Prepare directories ──────────────────────────────────────────────
    os.makedirs("models_checkpoint", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    # ── 2) Copy data if needed ───────────────────────────────────────────────
    # (You can replace this with shutil.copytree if you prefer)
    data_dir = os.path.join("dataset", dataset_name)
    if not os.path.isdir(data_dir):
        src = os.path.join("CAGE", "data", dataset_name)
        Util_class.folder_manage("dataset")           # create if missing
        Util_class.folder_manage(data_dir)            # create target
        # copy from your local CAGE checkout
        import shutil
        shutil.copytree(src, data_dir, dirs_exist_ok=True)

    # ── 3) File paths ────────────────────────────────────────────────────────
    edge_file    = os.path.join(data_dir, "in_edges.txt")
    feature_file = os.path.join(data_dir, "in_features.txt")
    label_file   = os.path.join(data_dir, "in_group.txt")

    # ── 4) Load ──────────────────────────────────────────────────────────────
    loader = LoadDataset(
        edge_file_name         = edge_file,
        attribute_file_name    = feature_file,
        label_file_name        = label_file,
        attribute_file_format  = "normalized_matrix",
        is_directed_graph      = False
    )
    net_adj = loader.get_structural_matrix()
    att_mat = loader.get_attribute_matrix()
    labels  = loader.get_labels()

    # ── 5) Batch generator ──────────────────────────────────────────────────
    batchGenerator = DataBatchGenerator(
        net               = net_adj,
        att               = att_mat,
        labels            = labels,
        batch_size        = 64,
        shuffle           = True,
        net_hadmard_coeff = 5.0,
        att_hadmard_coeff = 5.0
    )

    # ── 6) Model configuration ──────────────────────────────────────────────
    cfg = {
        "net_dim"             : loader.get_input_shape("net"),
        "net_layers_list"     : [
            {"type": "DENSE", "features":128, "act_funtion":"RELU", "bias":True}
        ],
        "net_latent_dim"      : 128,
        "att_dim"             : loader.get_input_shape("att"),
        "att_layers_list"     : [
            {"type":"KCOMP","ktop":200,"alpha_factor":3.0},
            {"type":"DENSE","features":128,"act_funtion":"RELU","bias":True}
        ],
        "att_latent_dim"      : 128,
        "loss_functions"      : {
            "net":[
                {"loss_name":"structur_proximity_1order","coef":1},
                {"loss_name":"structur_proximity_2order","coef":1}
            ],
            "att":[
                {"loss_name":"semantic_proximity_2order","coef":1},
                {"loss_name":"square_diff_embedding_proximity","coef":1}
            ]
        },
        "optimizator_net"     : {"opt_name":"adam","lr_rate":1e-4},
        "optimizator_att"     : {"opt_name":"adam","lr_rate":1e-3},
        "regularization_net"  : [{"reg_name":"L2","coeff":0.001}],
        "regularization_att"  : [{"reg_name":"L2","coeff":0.001}],
        "checkpoint_config"   : {
            "type":        ["best_model_loss","first_train","last_train"],
            "times":       20,
            "overwrite":   False,
            "path_file":   "models_checkpoint",
            "name_file":   f"CAGE_{dataset_name}_checkpoint",
            "path_not_exist":"create",
            "path_exist":  "use"
        },
        "model_name"          : f"CAGE_{dataset_name}_mlp",
        "training_config"     : "N>A"
    }

    # ── 7) Instantiate & train ───────────────────────────────────────────────
    model = GraphEModel(cfg)
    model.models_training(
        datagenerator  = batchGenerator,
        epochs         = {"net":10, "att":10},
        path_embedding = "embeddings/",
        loss_verbose   = False
    )

    # ── 8) Downstream eval ───────────────────────────────────────────────────
    pe = PerformanceEmbedding(model, embedding_name="att", node_label="node_label")

    # classification → DataFrame indexed by name_measure
    df_cls = pe.classification(repetitions=10)
    if "name_measure" in df_cls.columns:
        df_cls.set_index("name_measure", inplace=True)

    # clustering → only one row rand_score
    df_clu = pe.clusterization(repetitions=10)
    if "name_measure" in df_clu.columns:
        df_clu.set_index("name_measure", inplace=True)

    return df_cls, df_clu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    cls, clu = run_mlp(args.dataset, seed=args.seed)
    print(f"=== MLP Classification on {args.dataset} ===")
    print(cls)
    print(f"\n=== MLP Clustering (rand_score) on {args.dataset} ===")
    print(clu.loc[["rand_score"]])
