import torch
import torch.nn as nn
from einops import rearrange, repeat

torch.autograd.set_detect_anomaly(True)

    
class LSTM_MODEL(nn.Module):
    '''
        LSTM.
	'''
    def __init__(self, n_labels, input_dim, num_layers, bidirectional, latent_dim, dropout, device = torch.device("cpu")):

        super(LSTM_MODEL, self).__init__()

        self.device = device
        self.n_labels = n_labels
        if bidirectional:
            bi = 2
        else:
            bi = 1

        self.net = nn.LSTM(input_dim, latent_dim, batch_first=True, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional)
        
        self.linear = nn.Linear(in_features=latent_dim*bi, out_features=n_labels)
        self.act = nn.Sigmoid()
        
    def forward(self, data, time_steps=None, obj_t=None, pred_t=None):
        
        h, _ = self.net(data) # (batch, seq_len, emb_dim)
        
        h = h[:,-1,:]
        
        h = self.linear(h)
        
        h = self.act(h)
        
        return h