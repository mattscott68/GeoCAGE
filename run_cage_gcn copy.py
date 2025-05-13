#!/usr/bin/env python3
import os

os.makedirs("models_checkpoint", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
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
import shutil
import subprocess
import torch

from utils             import Util_class
from load_data         import LoadDataset
from batch_generator   import DataBatchGenerator
from graph_e_model_gcn import GraphEModel
from performance       import PerformanceEmbedding


def main():
    # ── 0) Torch print settings ──────────────────────────────────────────────
    torch.set_printoptions(edgeitems=64)

    # ── 2) Clone & copy CAGE/data → ./dataset ───────────────────────────────
    if not os.path.isdir("CAGE"):
        subprocess.run(["git", "clone", "https://github.com/MIND-Lab/CAGE.git"], check=False)
    for sub in os.listdir("CAGE/data"):
        src = os.path.join("CAGE", "data", sub)
        dst = os.path.join("dataset", sub)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    # ── 3) File paths (Cora) ────────────────────────────────────────────────
    dataset_name    = "cora"
    data_dir        = os.path.join("dataset", dataset_name)
    edge_file       = os.path.join(data_dir, "in_edges.txt")
    attribute_file  = os.path.join(data_dir, "in_features.txt")
    label_file      = os.path.join(data_dir, "in_group.txt")

    # ── 4) Load & export graph ──────────────────────────────────────────────
    loader = LoadDataset(
        edge_file_name        = edge_file,
        attribute_file_name   = attribute_file,
        label_file_name       = label_file,
        attribute_file_format = "normalized_matrix",
        is_directed_graph     = False
    )

    loader.export_graph(
        "models_checkpoint",               # folder to write into
        f"CAGE_{dataset_name}_graph"       # base name for the files
    )

    net_adj   = loader.get_structural_matrix()
    att_adj   = loader.get_attribute_matrix()
    labels    = loader.get_labels()
    print(f"Loaded dataset: {dataset_name}")

    # ── 5) Batch generator ──────────────────────────────────────────────────
    batchGenerator = DataBatchGenerator(
        net               = net_adj,
        att               = att_adj,
        labels            = labels,
        batch_size        = 64,
        shuffle           = True,
        net_hadmard_coeff = 5.0,
        att_hadmard_coeff = 5.0
    )

    # ── 6) Model configuration ──────────────────────────────────────────────
    CAGE_net_layers_list = [
        {"type":"DENSE","features":128,"act_funtion":"RELU","bias":True}
    ]
    CAGE_att_layers_list = [
        {"type":"KCOMP","ktop":200,"alpha_factor":3.0},
        {"type":"DENSE","features":128,"act_funtion":"RELU","bias":True}
    ]
    CAGE_loss_settings = {
        "net":[
            {"loss_name":"structur_proximity_1order","coef":1},
            {"loss_name":"structur_proximity_2order","coef":1}
        ],
        "att":[
            {"loss_name":"semantic_proximity_2order","coef":1},
            {"loss_name":"square_diff_embedding_proximity","coef":1}
        ]
    }
    CAGE_opt_net  = {"opt_name":"adam","lr_rate":1e-4}
    CAGE_opt_att  = {"opt_name":"adam","lr_rate":1e-3}
    CAGE_reg_net  = [{"reg_name":"L2","coeff":0.001}]
    CAGE_reg_att  = [{"reg_name":"L2","coeff":0.001}]
    CAGE_ckpt_cfg = {
        "type"        : ["best_model_loss","first_train","last_train"],
        "times"       : 20,
        "overwrite"   : False,
        "path_file"   : "models_checkpoint",
        "name_file"   : f"CAGE_{dataset_name}_checkpoint",
        "path_not_exist":"create",
        "path_exist"  :"use"
    }

    cfg = {
        "net_dim"             : loader.get_input_shape("net"),
        "net_layers_list"     : CAGE_net_layers_list,
        "net_latent_dim"      : 128,
        "att_dim"             : loader.get_input_shape("att"),
        "att_layers_list"     : CAGE_att_layers_list,
        "att_latent_dim"      : 128,
        "loss_functions"      : CAGE_loss_settings,
        "optimizator_net"     : CAGE_opt_net,
        "optimizator_att"     : CAGE_opt_att,
        "regularization_net"  : CAGE_reg_net,
        "regularization_att"  : CAGE_reg_att,
        "checkpoint_config"   : CAGE_ckpt_cfg,
        "model_name"          : f"CAGE_{dataset_name}_gcn",
        "training_config"     : "N>A",
        "net_adj"             : net_adj
    }

    # ── 7) Instantiate & train ───────────────────────────────────────────────
    model = GraphEModel(cfg)
    epochs = {"att":10, "net":10}
    DAGE_values = model.models_training(
        datagenerator = batchGenerator,
        epochs        = epochs,
        loss_verbose  = False,
        path_embedding= "models_checkpoint/"
    )

    # ── 8) (Optional) Downstream evaluation ─────────────────────────────────
    perf       = PerformanceEmbedding(model, embedding_name="att")
    df_class   = perf.classification(repetitions=10)
    df_cluster = perf.clusterization(repetitions=10)
    print("=== GCN Classification ===\n", df_class)
    print("=== GCN Clustering (rand_score) ===\n", df_cluster)

if __name__ == "__main__":
    main()
