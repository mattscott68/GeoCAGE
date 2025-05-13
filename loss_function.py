import torch
from torch import nn
import torch.nn.functional as F

all_a = None
all_b = None
class LossFunction(nn.Module):
    """
    loss_functions : list of dictionary:
            loss_name : loss function name,
            coef : coefficent to totla loss function
    matrix_values : dictionary
            net : structure matrix
                y_true : grountruth matrix
                y_late : embedding matrix
                y_pred : predict matrix
             att : semantical matrix
                y_true : grountruth  matrix
                y_late : embedding matrix
                y_pred : predict matrix
    """
    def __init__(self, loss_functions,matrix_values):
      self.loss_functions = loss_functions
      self.matrix_values = matrix_values

    def loss_computate(self,verbose=False):

        loss_total = torch.zeros(1)
        for loss_function in self.loss_functions:
            loss_name = loss_function['loss_name']

            coef = loss_function['coef']
            if isinstance(coef, float) or isinstance(coef, int):
                if loss_name == "structur_proximity_1order":
                    _val = self.structur_proximity_1order( self.matrix_values["net"]["y_late"], self.matrix_values["net"]["y_adj"],self.matrix_values["net"])
                    loss_total.add_(_val.mul(coef))
                elif loss_name == "semantic_proximity_1order":
                    _val = self.semantic_proximity_1order( self.matrix_values["att"]["y_late"], self.matrix_values["net"]["y_adj"])
                    loss_total.add_(_val.mul(coef))
                elif loss_name == "structur_proximity_2order":
                    _val = self.structur_proximity_2order( self.matrix_values["net"]["y_pred"], self.matrix_values["net"]["y_true"], self.matrix_values["net"]["B_param"])
                    loss_total.add_(_val.mul(coef))
                elif loss_name == "semantic_proximity_2order":
                    _val = self.semantic_proximity_2order( self.matrix_values["att"]["y_pred"], self.matrix_values["att"]["y_true"], self.matrix_values["att"]["B_param"])
                    loss_total.add_(_val.mul(coef))
                elif loss_name == "consisency_proximity":
                    _val = self.consisency_proximity( self.matrix_values["net"]["y_late"], self.matrix_values["att"]["y_late"])
                    loss_total.add_(_val.mul(coef))
                elif loss_name == "consisency_compl_proximity":
                    _val = self.consisency_compl_proximity( self.matrix_values["net"]["y_late"], self.matrix_values["att"]["y_late"])
                    loss_total.add_(_val.mul(coef))
                elif loss_name == "square_diff_embedding_proximity":
                    _val = self.square_diff_embedding_proximity( self.matrix_values["net"]["y_late"], self.matrix_values["att"]["y_late"], self.matrix_values["net"]["y_adj"])
                    loss_total.add_(_val.mul(coef))
                elif loss_name == "laplacian_smoothness":
                    _val = self.laplacian_smoothness(
                        self.matrix_values["net"]["y_late"],
                        self.matrix_values["net"]["edge_index"],
                        self.matrix_values["net"].get("edge_weight", None)
                    )
                    loss_total = loss_total + _val.mul(coef)

                elif loss_name == "reconstruction_mse":
                    # adjacency reconstruction loss
                    _val = self.reconstruction_mse(
                        self.matrix_values["net"]["y_pred"],
                        self.matrix_values["net"]["y_true"]
                    )
                    loss_total = loss_total + _val.mul(coef)

                elif loss_name == "alignment_proximity":
                    # attribute–structure alignment regularizer
                    _val = self.alignment_proximity(
                        self.matrix_values["net"]["y_late"],
                        self.matrix_values["att"]["y_late"]
                    )
                    loss_total = loss_total + _val.mul(coef)

                else:
                    raise LossFunction_Exception_FuntionNotExist(loss_name)
            else:
                raise LossFunction_Exception_Coeff(loss_name, coef)
            if verbose:
                print("\t",loss_name,"\t->\t",_val,"\tTOT:\t",loss_total)
        if verbose:
            print("----\tTOTAL:\t",loss_total,"\t\t----\n")
        return loss_total


    def structur_proximity_1order(self, hs_emb, w_matrix,oth = None):
        """
        hs_emb : embedding matrix
        w_matrix : structural adjacency matrix
        return a tensor with value 0
        """
        sigmoid_argument = torch.matmul(hs_emb,torch.transpose(hs_emb,0,1))
        labels_1 = w_matrix + torch.eye(w_matrix.size()[0])
        cross_E1 = self.__sigmoid_cross_entropy_with_logits(labels= labels_1, logits= sigmoid_argument,inp=sigmoid_argument,oth=oth)
        labels_2 = torch.ones_like(torch.diag(sigmoid_argument))
        logits_2 = torch.diag(sigmoid_argument)
        cross_E2 = self.__sigmoid_cross_entropy_with_logits(labels= labels_2, logits= logits_2,oth=oth)
        cross_All = cross_E1 - cross_E2
        return torch.mean(cross_All)

    def semantic_proximity_1order(self, hs_emb, w_matrix):
        """
        hs_emb : embedding matrix
        w_matrix : structural adjacency matrix
        return a tensor with value 0
        """
        sigmoid_argument = torch.matmul(hs_emb,torch.transpose(hs_emb,0,1))
        labels_1 = w_matrix + torch.eye(w_matrix.size()[0])
        cross_E1 = self.__sigmoid_cross_entropy_with_logits(labels= labels_1, logits= sigmoid_argument)
        labels_2 = torch.ones_like(torch.diag(sigmoid_argument))
        logits_2 = torch.diag(sigmoid_argument)
        cross_E2 = self.__sigmoid_cross_entropy_with_logits(labels= labels_2, logits= logits_2)
        cross_All = cross_E1 - cross_E2
        return torch.mean(cross_All)

    def structur_proximity_2order(self, y_pred, y_true, b_param):
        # single Frobenius‐style norm over all edge‐predictions
        return torch.norm(((y_pred - y_true) * b_param).pow(2), p=2)

    def semantic_proximity_2order(self, ys_true, ys_pred, b_param):
        """
        ys_true : vector of items where each item is a groundtruth matrix
        ys_pred : vector of items where each item is a prediction matrix
        return the sum of 2nd proximity of 2 matrix
        """
        loss_secondary = 0

        for i, y_true in enumerate(ys_true):
            y_pred = ys_pred[i]
            loss_secondary_item = torch.norm(torch.square(torch.sub(y_pred,y_true,alpha=1) * b_param), p=2)
            loss_secondary += loss_secondary_item
        return loss_secondary

    def consisency_proximity(self, hs_net, hs_att):
        """
        hs_net : matrix embedding structure
        hs_att : matrix embedding attribute
        return the consisency proximity value
        """
        loss_secondary = 0

        for i, h_net in enumerate(hs_net):
            h_att = hs_att[i]
            loss_secondary_item = torch.norm(torch.square(torch.sub(h_att,h_net,alpha=1)), p=2)
            loss_secondary += loss_secondary_item
        return loss_secondary


    def consisency_compl_proximity(self, hs_net, hs_att):
        """
        hs_net : matrix embedding structure
        hs_att : matrix embedding attribute
        return the consisency proximity value
        """

        logits = torch.sum(torch.multiply(hs_net, hs_att), dim=1)
        labels = torch.ones_like(logits)
        cross_All = self.__sigmoid_cross_entropy_with_logits(labels= labels, logits= logits)
        return torch.mean(cross_All)
    
    def laplacian_smoothness(self, Z, edge_index, edge_weight=None):
        # Tr(Zᵀ L Z) = ½ Σ_{(i,j)} w_{ij} ||z_i - z_j||²
        row, col = edge_index        # both are [E] long
        diffs   = Z[row] - Z[col]    # [E, d]
        sqdist  = (diffs * diffs).sum(dim=1)  # [E]
        if edge_weight is None:
            return 0.5 * sqdist.mean()
        else:
            return 0.5 * (edge_weight * sqdist).sum()
    
    def reconstruction_mse(self, y_pred, y_true):
        """Simple MSE between decoded edges and ground‑truth adjacency."""
        return F.mse_loss(y_pred, y_true)

    def alignment_proximity(self, Z_S, H_A):
        """
        ∑_{i,j} (‖h_i^A - h_j^A‖² - ‖z_i^S - z_j^S‖²)²
        """
        # squared‑distance matrices
        D_A = torch.cdist(H_A, H_A, p=2).pow(2)
        D_S = torch.cdist(Z_S, Z_S, p=2).pow(2)
        return torch.sum((D_A - D_S).pow(2))

    def __softmax_cross_entropy_with_logits(self, labels, logits):
        _cross_entropy = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1)
        return _cross_entropy

    def __sigmoid_cross_entropy_with_logits(self, labels, logits,inp=None, oth=None):
        eps = 1e-12
        _cross_entropy_a = (labels * -torch.log(torch.sigmoid(logits) + eps))
        _cross_entropy_b = (1 - labels) * - torch.log(1 - torch.sigmoid(logits) + eps)
        _cross_entropy = _cross_entropy_a + _cross_entropy_b
        return _cross_entropy

    def square_diff_embedding_proximity(self, hs_net, hs_att, w_matrix):
        """
        ys_true : vector of items where each item is a groundtruth matrix
        ys_pred : vector of items where each item is a prediction matrix
        return the sum of 2nd proximity of 2 matrix
        """

        struct_proximity = self.structur_proximity_1order(hs_net, w_matrix)
        attrib_proximity = self.semantic_proximity_1order(hs_att, w_matrix)
        loss_square = torch.square(attrib_proximity + torch.neg(struct_proximity))
        return loss_square



class LossFunction_Exception_Coeff(Exception):
      """Exception raised for error if coeff is not int or float"""

      def __init__(self, loss_name, value):
          self.value = value
          self.loss_name = loss_name

      def __str__(self):
          return f'Loss "{self.loss_name}" coefficent should be a float or int but receive a {type(self.value)}.'

class LossFunction_Exception_FuntionNotExist(Exception):
      """Exception raised for error if coeff is not int or float"""

      def __init__(self, loss_name):
          self.loss_name = loss_name

      def __str__(self):
          return f'Loss "{self.loss_name}" not exist.'