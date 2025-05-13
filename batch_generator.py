import numpy as np
import torch

class DataBatchGenerator():

    def __init__(self, net, att, labels, batch_size, shuffle, net_hadmard_coeff, att_hadmard_coeff):
        self.net = net
        self.att = att
        N = self.net.shape[0]
        M, F = self.att.shape
        if M < N:
            # pad missing rows with zeros
            pad = np.zeros((N - M, F), dtype=self.att.dtype)
            self.att = np.vstack([self.att, pad])
        elif M > N:
            # chop off any extra rows
            self.att = self.att[:N, :]
            
        # 2) pad / chop labels to match N
        #    If you have “unknown” for padded nodes, you might use -1,
        #    or repeat the last label, or fill with a dummy class.
        L = len(labels)
        if L < N:
            # pad with -1 (or choose another sentinel)
            pad_lbls = np.full((N - L,), -1, dtype=labels.dtype)
            self.labels = np.concatenate([labels, pad_lbls])
        else:
            # chop off extras
            self.labels = labels[:N]

        # now self.att.shape[0] == self.labels.shape[0] == N
        self.number_of_samples = self.att.shape[0]   # ← use the padded / truncated version
        self.batch_size = batch_size
        self.number_of_batches = self.number_of_samples // batch_size
        self.shuffle = shuffle
        self.net_hadmard_coeff = net_hadmard_coeff
        self.att_hadmard_coeff = att_hadmard_coeff

    def generate(self):
        sample_index = np.arange(self.net.shape[0])

        counter = 0
        if self.shuffle:
            np.random.shuffle(sample_index)

        while (counter*self.batch_size < self.number_of_samples):
            start_samples_index = self.batch_size * counter
            end_samples_index = self.batch_size * (counter + 1)

            #list of samples's index
            samples_index = sample_index[start_samples_index : end_samples_index]

            #submatrix of W and A, cut for sample index
            net_batch = self.net[samples_index, :]
            att_batch = self.att[samples_index, :]
            net_batch_adj = self.net[samples_index, :][:, samples_index]
            node_label = self.labels[samples_index]
            node_index = samples_index

            # B_net and B_att param of hadmard operation
            B_net = np.ones(net_batch.shape)
            B_net[net_batch != 0] = self.net_hadmard_coeff

            B_att = np.ones(att_batch.shape)
            B_att[att_batch != 0] = self.att_hadmard_coeff

            # trasform np array to tensor
            net_batch_tensor = torch.from_numpy(net_batch).float()
            att_batch_tensor = torch.from_numpy(att_batch).float()
            net_batch_adj_tensor = torch.from_numpy(net_batch_adj).float()
            B_net_tensor = torch.from_numpy(B_net).float()
            B_att_tensor = torch.from_numpy(B_att).float()

            inputs = [net_batch_tensor, att_batch_tensor, net_batch_adj_tensor]
            B_params = [B_net_tensor, B_att_tensor]
            batch_info = [node_index, node_label]

            # feed the fit() function with new data
            yield inputs, B_params, batch_info
            counter += 1