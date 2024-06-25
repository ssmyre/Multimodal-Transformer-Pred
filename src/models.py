import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modules.transformer import TransformerEncoder
# from modules.transformer import TransformerDecoder

X_A_SHAPE = (128,128)
"""Processed spectrogram shape: ``[freq_bins, time_bins]``"""
X_A_DIM = np.prod(X_A_SHAPE)
"""Processed spectrogram dimension: ``freq_bins * time_bins``"""

X_V_SHAPE = (480,640)
"""Processed video shape: ``[freq_bins, time_bins]``"""
X_V_DIM = np.prod(X_V_SHAPE)
"""Processed video dimension: ``freq_bins * time_bins``"""

X_L_SHAPE = (1,1)
X_L_DIM = np.prod(X_L_SHAPE)


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # print(last_hs.shape)
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        # print(last_hs_proj.shape) # size []

        output = self.out_layer(last_hs_proj)
        return output, last_hs  #LOOK UP WHAT THIS IS?
    
class MULTModelPred(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModelPred, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 32, 32, 32
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        self.combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            self.combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            self.combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)


        # Projection layers
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    

    def decode_l(self, z, rec_dim):
        out_layer_l = nn.Linear(self.combined_dim, rec_dim)
        output = out_layer_l(z)
        return output
    
    def decode_a(self, z, rec_dim):
        #power of 2 and 
        #separate decoders for each modality
        self.fc1 = nn.Linear(self.combined_dim,192)
        self.fc2 = nn.Linear(192,768)
        self.fc3 = nn.Linear(768,3072)
        self.fc4 = nn.Linear(3072,24576)
        self.convt1 = nn.ConvTranspose2d(96,48,3,1,padding=1) #collapse across filters
        self.convt2 = nn.ConvTranspose2d(48,48,3,2,padding=1,output_padding=1) # changing size of output image
        self.convt3 = nn.ConvTranspose2d(48,24,3,1,padding=1)
        self.convt4 = nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1)
        self.convt5 = nn.ConvTranspose2d(24,8,3,1,padding=1)
        self.convt6 = nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1)
        self.convt7 = nn.ConvTranspose2d(8,1,3,1,padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(48)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(24)
        self.bn5 = nn.BatchNorm2d(24)
        self.bn6 = nn.BatchNorm2d(8)
        self.bn7 = nn.BatchNorm2d(8)

        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = z.view(-1,96,16,16)
        z = F.relu(self.convt1(self.bn1(z)))
        z = F.relu(self.convt2(self.bn2(z)))
        z = F.relu(self.convt3(self.bn3(z)))
        z = F.relu(self.convt4(self.bn4(z)))
        z = F.relu(self.convt5(self.bn5(z)))
        z = F.relu(self.convt6(self.bn6(z)))
        z = self.convt7(self.bn7(z))
        return z.view(-1, X_A_DIM)
    
    def decode_v(self, z, rec_dim):
        #separate decoders for each modality
        self.fc5 = nn.Linear(self.combined_dim,192)
        self.fc6 = nn.Linear(192,960)
        self.fc7 = nn.Linear(960,4800)
        self.fc8 = nn.Linear(4800,14400)
        self.fc9 = nn.Linear(14400,28800)
        self.convt8 = nn.ConvTranspose2d(96,48,3,1,padding=1) #collapse across filters
        self.convt9 = nn.ConvTranspose2d(48,48,3,2,padding=1,output_padding=1) # changing size of output image
        self.convt10 = nn.ConvTranspose2d(48,32,3,1,padding=1)
        self.convt11 = nn.ConvTranspose2d(32,32,3,2,padding=1,output_padding=1) # changing size of output image
        self.convt12 = nn.ConvTranspose2d(32,24,3,1,padding=1)        
        self.convt13 = nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1)
        self.convt14 = nn.ConvTranspose2d(24,16,3,1,padding=1)
        self.convt15 = nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1)
        self.convt16 = nn.ConvTranspose2d(16,8,3,1,padding=1)
        self.convt17 = nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1)
        self.convt18 = nn.ConvTranspose2d(8,1,3,1,padding=1)
        self.bn8 = nn.BatchNorm2d(96)
        self.bn9 = nn.BatchNorm2d(48)
        self.bn10 = nn.BatchNorm2d(48)
        self.bn11 = nn.BatchNorm2d(32)
        self.bn12 = nn.BatchNorm2d(32)        
        self.bn13 = nn.BatchNorm2d(24)
        self.bn14 = nn.BatchNorm2d(24)
        self.bn15 = nn.BatchNorm2d(16)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn17 = nn.BatchNorm2d(8)
        self.bn18 = nn.BatchNorm2d(8)

        z = F.relu(self.fc5(z))
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = F.relu(self.fc8(z))
        z = F.relu(self.fc9(z))
        z = z.view(-1,96,20,15)
        z = F.relu(self.convt8(self.bn8(z)))
        z = F.relu(self.convt9(self.bn9(z)))
        z = F.relu(self.convt10(self.bn10(z)))
        z = F.relu(self.convt11(self.bn11(z)))
        z = F.relu(self.convt12(self.bn12(z)))
        z = F.relu(self.convt13(self.bn13(z)))
        z = F.relu(self.convt14(self.bn14(z)))
        z = F.relu(self.convt15(self.bn15(z)))
        z = F.relu(self.convt16(self.bn16(z)))
        z = F.relu(self.convt17(self.bn17(z)))        
        z = self.convt18(self.bn18(z))
        return z.view(-1, X_V_DIM)


    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction       Dimension(batch_size, embed dimension)

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs


        #Use last_h_l   last_h_a    last_h_v to decode image for next time step
        #normalize across modalities using mean squared error
        #reconstruction errors for every modality        
        #use one last_hs_proj as input to decode and output different size

        rec_l =  self.decode_l(last_hs_proj,self.orig_d_l) #reconstruct original text
        rec_a =  self.decode_a(last_hs_proj,self.orig_d_a) #reconstruct original audio image
        rec_v =  self.decode_v(last_hs_proj,self.orig_d_v) #reconstruct original video image
        # output = self.out_layer(last_hs_proj)
        return rec_l, rec_a, rec_v, last_hs
