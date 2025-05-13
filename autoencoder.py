import torch
from torch import nn, optim
import torch_geometric
from kcompetitive import KCompetitiveLayer

class AutoEncoder(nn.Module):
    """
    param dim: original dimension
    param layers_list : sequential sort list of dict : Each item  have a value for "type", as:
                                          DENSE -  hidden layers, with: "features" is dimention of features in output, "act_funtion" is the relative activation funcion,'bias' boolean
                                          DROP  -  dropout, with: "prob" is the percentaul of neuro drop
                                          KCOMP -  kcompetitivelayer,, with "ktop":int #of active neurons at end of computation, "alpha_factor":float coefficent
    param latent_dim : dimension of latent space
    last_isSigm : boolean, True if last activation function of decoder is a sigmoid
    return : autoencoder model
    """

    def __init__(self, dim, layers_list, latent_dim,last_isSigm= True):

        super().__init__()
        self.activation = {}
        self.encoder_list=[]
        self.decoder_list=[]

        last_dim = dim
        for i,layer in enumerate(layers_list):
            if layer['type'] == "DROP":
                prob = layer['prob']
                if isinstance(prob, float) and 0 <= prob <= 1:
                    self.encoder_list.append(torch.nn.Dropout(p=prob))
                    self.decoder_list.insert(0,torch.nn.Dropout(p=prob))
                else:
                    raise AutoEncoder_Exception_DropoutProb(prob)

            elif layer['type'] == "DENSE":
                self.encoder_list.append(torch.nn.Linear(in_features=last_dim, out_features=layer['features'], bias=layer['bias']))
                if layer['act_funtion'] == "RELU":
                    self.encoder_list.append(torch.nn.ReLU())
                    decoder_layer_funact = torch.nn.ReLU()
                elif layer['act_funtion'] == "SIGM":
                    self.encoder_list.append(torch.nn.Sigmoid())
                    decoder_layer_funact = torch.nn.Sigmoid()
                else:
                    raise AutoEncoder_Exception_ActivationFunction(layer['act_funtion'])

                if i == 0 and last_isSigm:
                  decoder_layer_funact = torch.nn.Sigmoid()
                self.decoder_list.insert(0,decoder_layer_funact)
                self.decoder_list.insert(0,torch.nn.Linear(in_features=layer['features'], out_features=last_dim, bias=layer['bias']))
                last_dim = layer['features']
            elif layer['type'] == "KCOMP":
                competitiveLayers = KCompetitiveLayer(layer['ktop'], layer['alpha_factor'])
                self.encoder_list.append(competitiveLayers)
            else:
                raise AutoEncoder_Exception_Type(layer['type'])



        if last_dim != latent_dim:
            raise AutoEncoder_Exception_LatentSpace(last_dim,latent_dim)
        self.encoder = nn.Sequential(*self.encoder_list)
        self.decoder = nn.Sequential(*self.decoder_list)


    def forward(self,x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent)
        return {"x_input":x,"x_latent":x_latent,"x_output":x_hat}


class AutoEncoder_Exception_Type(Exception):
      """Exception raised for errors in list of layers type"""

      def __init__(self, value):
          self.value = value

      def __str__(self):
          return f'{self.value} : type layer not recognized: it should be a hidden layer linear (DENSE) or dropout layer (DROP).'


class AutoEncoder_Exception_DropoutProb(Exception):
      """Exception raised for errors in list of layers type"""

      def __init__(self, value):
          self.value = value

      def __str__(self):
          if isinstance(self.value, float):
              return f'Dropout should have probability param in range 0 to 1, but receive {self.value}.'
          else:
              return f'Dropout should be a float but receive a {type(self.value)}.'



class AutoEncoder_Exception_ActivationFunction(Exception):
      """Exception raised for errors in activation function type"""

      def __init__(self, value):
          self.value = value

      def __str__(self):
          return f'{self.value} : activation function not recognized: it should be a relu function (RELU), a sigmoid funcion (SIGM).'



class AutoEncoder_Exception_LatentSpace(Exception):
      """Exception raised for errors in list of layers type: last layer in list haven't the latent space dimention"""

      def __init__(self, last_dim,latent_dim):
          self.last_dim = last_dim
          self.latent_dim = latent_dim

      def __str__(self):
          return f'Last layer have {self.last_dim} output dimention but latent space should be {self.latent_dim}.'
      
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, APPNP  # added APPNP import

class GraphAutoencoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 K: int = 10,
                 alpha: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        # 1) Initial linear → latent projection (skip)
        self.lin_proj = nn.Linear(in_dim, latent_dim, bias=False)

        # 2) GCN layers + BatchNorm
        self.conv1 = GCNConv(latent_dim, hidden_dim, normalize=True)
        self.bn1   = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, latent_dim, normalize=True)
        self.bn2   = nn.BatchNorm1d(latent_dim)

        # 3) APPNP propagation
        self.prop   = APPNP(K=K, alpha=alpha)

        # 4) Dropout
        self.dropout = nn.Dropout(p=dropout)

    def encode(self, x, edge_index):
        # -- skip projection
        z0 = self.lin_proj(x)                     # [N, d_latent]

        # -- first GCN block
        h1 = self.conv1(z0, edge_index)           # [N, hidden]
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # -- second GCN block
        h2 = self.conv2(h1, edge_index)           # [N, d_latent]
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)

        # -- APPNP propagation + residual skip
        z_refined = self.prop(h2, edge_index)     # [N, d_latent]
        z_final   = z_refined + z0                # residual: APPNP + skip

        return z_final

    def decode(self, z, edge_index):
        # inner–product decoder
        row, col = edge_index
        return torch.sigmoid((z[row] * z[col]).sum(dim=1))

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index), z

