import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils import cal_cos_similarity
import torch.nn.functional as F


class FACTORMODEL(nn.Module):
    def __init__(self, d_feat=6, hidden_size=32, num_layers=2, dropout=0.0, kernel_size = 3, dilations=[1,2,5], num_days=60, K=3, ):
        super().__init__()
        
        self.n_models = len(dilations)
        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.num_days = num_days
        # hidden_feat = 64
        hidden_feat = 128
        self.is_Hist = K>0
        
        
        multimodel = [] 
        for i in range(self.n_models):
          multimodel.append(CONVENC(d_feat = d_feat, 
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    kernel_size = kernel_size,
                                    dilation = dilations[i]))
                                    
              
        self.multimodel = torch.nn.ModuleList(multimodel)
        
        self.hid2source = nn.Linear(hidden_size, d_feat)
        self.attdec = ATTDEC(d_feat= d_feat, hidden_size = hidden_size, K=K)
        self.predictrnn = PREDICTRNN(hidden_size = hidden_size, hidden_feat = hidden_feat, n_models = self.n_models, stride2 = dilations)
        
        self.tanh = nn.Tanh()
        
        if self.is_Hist:
          self.feathist = FEATHIST(hidden_size = hidden_feat, K = K)
        else:
          self.fc_out = nn.Linear(hidden_feat, 1)
        
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
            
    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x.reshape(-1, self.d_feat), x.reshape(-1, self.d_feat), reduction='mean')
        kldivergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return [recon_loss, kldivergence] 
        
    
    # the loss for the AR
    def seq_rec_loss_2(self, alpha, x, ps_hidden, slc_days):

      is_slc = slc_days[-1]
      for i in range(self.n_models-1):
        is_slc = np.intersect1d(is_slc,slc_days[i])
      if is_slc.size==0:
        loss_seq = 0.
      else:

        x = x[:,is_slc,:]
        num_days = len(is_slc)
        x_hidden = []
        bs = x.size()[0]
        alpha = alpha.unsqueeze(1)
        alpha = alpha.expand(bs,num_days,self.n_models)
        alpha = alpha.unsqueeze(3)
        alpha = alpha.reshape(-1,self.n_models,1)
        for i in range(self.n_models):
          d_slc = [list(slc_days[i]).index(j) for j in is_slc]
          p_hidden = ps_hidden[i]
          p_hidden = p_hidden[:,d_slc,:]
        
          recon_x = self.hid2source(p_hidden)
          x_hidden.append(recon_x.unsqueeze(3))
        x_hidden = torch.cat(x_hidden,3)
        x_hidden = torch.bmm(x_hidden.reshape(-1,self.d_feat,self.n_models), alpha).squeeze()
        loss_seq = F.mse_loss(x_hidden.reshape(-1, self.d_feat), x.reshape(-1, self.d_feat), reduction='mean')
      return loss_seq
      
      
    def permute_dims(self, z):
        assert z.dim() == 2

        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)

        return torch.cat(perm_z, 1)
    
    def forward(self, x, concept_matrix, market_value, no_dec=False, fVAE=False):
    
        device = torch.device(torch.get_device(x))
        x_hidden_mu = []
        x_hidden_sigma = []
        x_hidden_sample = []
        x = x.reshape(len(x), self.d_feat, -1)
        
        time_length = x.size()[-1]
        x = x[:,:,(time_length-self.num_days):]
        
        for i in range(self.n_models):

          x_hidden = self.multimodel[i](x)
          x_hidden_mu.append(x_hidden[:,:,:self.hidden_size])
          x_hidden_sigma.append(x_hidden[:,:,self.hidden_size:])
          x_sample = self.latent_sample(x_hidden[:,:,:self.hidden_size],x_hidden[:,:,self.hidden_size:])

          x_hidden_sample.append(x_sample.unsqueeze(3)) 
        
        
        x_hidden_sample = torch.cat(x_hidden_sample, 3)
        x_hidden_sigma = torch.cat(x_hidden_sigma, 2)
        x_hidden_mu = torch.cat(x_hidden_mu, 2)
        
        x_hidden_sample = x_hidden_sample.permute(0,1,3,2)
        
        if no_dec:
          z_prime = self.permute_dims(x_hidden_sample.reshape(-1,self.hidden_size))
          return z_prime
        
        recon_x = self.hid2source(x_hidden_sample)
        x = x.permute(0, 2, 1)
        recon_x, alpha = self.attdec(recon_x, x) 
        
        loss_vae = self.vae_loss(recon_x, x, x_hidden_mu, x_hidden_sigma)
        
        x_hidden,ps_hidden,slc_days = self.predictrnn(x_hidden_sample)
        loss_seq = self.seq_rec_loss_2(alpha, x, ps_hidden, slc_days)

        
        alpha = alpha.unsqueeze(2) # (N, 3 ,1)
        
        x_hidden = torch.bmm(x_hidden, alpha).squeeze()

        if self.is_Hist:
          pred_all = self.feathist(x_hidden, concept_matrix, market_value) 
        else:
          pred_all = self.fc_out(x_hidden).squeeze()
          
        if self.training and fVAE:
          return pred_all, loss_vae+[loss_seq], x_hidden_sample.reshape(-1,self.hidden_size)
        else:
          return pred_all, loss_vae+[loss_seq]

# encoders with different sampling rates (dilation rates)
class CONVENC(nn.Module):
    def __init__(self, d_feat=6, hidden_size=32, num_layers=2, kernel_size = 3, dropout=0.0, dilation = 1):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        multiconv = [] 
        for i in range(num_layers):
          if i == 0:
            multiconv.append(nn.Conv1d(d_feat, hidden_size, kernel_size, dilation=dilation,padding=0))
          else:
            multiconv.append(nn.Conv1d(hidden_size, hidden_size, kernel_size, dilation=dilation,padding=0))
          
        self.multiconv = nn.ModuleList(multiconv)
        self.residual = nn.Linear(d_feat, hidden_size)
        self.fc_musig = nn.Linear(hidden_size, hidden_size*2)
        self.leaky_relu = nn.LeakyReLU()
        
        nn.init.xavier_uniform_(self.fc_musig.weight)
        nn.init.xavier_uniform_(self.residual.weight)
        for i in range(num_layers):
          nn.init.xavier_uniform_(self.multiconv[i].weight)


    def forward(self, x_hidden):

        p1d = ((self.kernel_size-1)*self.dilation, 0) # pad 0 before #days to keep the output of the conv1d unchanged
        
        
        
        x_res = self.residual(x_hidden.permute(0,2,1))

        p1d_res = (0, self.hidden_size)
        x_res = F.pad(x_res, p1d_res, "constant", 0) 
        
        
        for i in range(self.num_layers):
          x_hidden = F.pad(x_hidden, p1d, "constant", 0)
          # print(x_hidden[0,0,:])
          x_hidden = self.multiconv[i](x_hidden) 
          x_hidden = self.leaky_relu(x_hidden)
          # print(x_hidden.size())
        
        x_hidden = x_hidden.permute(0, 2, 1)
        x_hidden = self.fc_musig(x_hidden)

        return x_hidden+x_res 

# the attention operation of the decoder
class ATTDEC(nn.Module):
    def __init__(self, d_feat = 6, hidden_size = 32, K = 0):
      super().__init__()
      self.fc_q = nn.Linear(d_feat, hidden_size)
      self.fc_k = nn.Linear(d_feat, hidden_size)
      self.hidden_size = hidden_size
      self.d_feat = d_feat
      self.K = K

    def forward(self, x, x_ref):

      bs = x.size()[0]
      n_signal = x.size()[2]


      x_hid = self.fc_q(x)
      x_ref = self.fc_k(x_ref) 
      
      x_hid = x_hid.reshape(-1, n_signal, self.hidden_size)
      x_ref = x_ref.reshape(-1, self.hidden_size)
      
      x = x.reshape(-1, n_signal, self.d_feat) 
      
      alpha = torch.bmm(x_hid,x_ref.unsqueeze(2)).squeeze()
      alpha = F.softmax(alpha, 1)
      if self.K==0:
        alpha = alpha+1
      
      
      x = torch.bmm(x.permute(0,2,1),alpha.unsqueeze(2))
      
      alpha = alpha.reshape(bs, -1, n_signal)# (N, num_days, 3)
      alpha = torch.mean(alpha,1)# (N, 3)
      
      return x.reshape(bs, -1, self.d_feat), alpha

# the decoder and the predictor
class PREDICTRNN(nn.Module):
    def __init__(self, hidden_size=64, hidden_feat = 64, dropout=0.0, n_models=1, stride2 = [1], base_model="GRU"):
        super().__init__()

        # self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.n_models = n_models
        self.stride2 = stride2
        self.hidden_feat = hidden_feat

        multimodel = [] 
        for i in range(self.n_models):
          # the AR
          multimodel.append(nn.GRU(input_size=hidden_size, 
                                    hidden_size = hidden_size,
                                    num_layers = 1,
                                    batch_first = True,
                                    dropout = dropout))
          # predictor of the stock return representaion
          multimodel.append(nn.GRU(input_size=hidden_size*2, 
                                    hidden_size = hidden_feat,
                                    num_layers = 1,
                                    batch_first = True,
                                    dropout = dropout))
                                    
              
        self.multimodel = torch.nn.ModuleList(multimodel)


    def forward(self, x):

        bs = x.size()[0]
        
        x_hidden = []
        ps_hidden = []
        slc_days = []
        for i in range(self.n_models):
          x_hidden_temp = x[:,:,i,:] 

          stride2 = self.stride2[i]
          slc_day = range(x_hidden_temp.size()[1], -1, -stride2)
          
          slc_day = slc_day[1:]
          x_hidden_1 = x_hidden_temp[:,slc_day[::-1],:]
          
          p_hidden, _ = self.multimodel[2*i](x_hidden_1) 
          ps_hidden.append(p_hidden[:, :-1, :])
          slc_days.append(np.array(slc_day[::-1])[:-1]+1)
          
          
          p_hidden = p_hidden[:, -1, :].reshape(bs,1,self.hidden_size) 
          if not (x_hidden_temp.size()[1] % 2):
            x_hidden_temp = x_hidden_temp[:, 1:, :]
          x_hidden_temp = torch.cat([x_hidden_temp,p_hidden],1) 
          
          slc_day = range(x_hidden_temp.size()[1]-1, 0, -stride2)
          
          x_hidden_2 = x_hidden_temp[:,slc_day[::-1],:]

          slc_day = range(x_hidden_temp.size()[1]-2, -1, -stride2)
          x_hidden_3 = x_hidden_temp[:,slc_day[::-1],:]
          
          
          x_hidden_temp = torch.cat([x_hidden_2,x_hidden_3],2)
          x_hidden_temp, _ = self.multimodel[2*i+1](x_hidden_temp) 
          
          x_hidden.append(x_hidden_temp[:, -1, :].unsqueeze(2))
        
        x_hidden = torch.cat(x_hidden, 2)
          
        return x_hidden, ps_hidden, slc_days

# the HIST module for prediction
class FEATHIST(nn.Module):
    def __init__(self, hidden_size=64, dropout=0.0, K = 3):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc_ps = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps.weight)
        self.fc_hs = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs.weight)

        self.fc_ps_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_fore.weight)
        self.fc_hs_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_fore.weight)

        self.fc_ps_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_back.weight)
        self.fc_hs_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim = 0)
        self.softmax_t2s = torch.nn.Softmax(dim = 1)
        
        self.fc_out_ps = nn.Linear(hidden_size, 1)
        self.fc_out_hs = nn.Linear(hidden_size, 1)
        self.fc_out_indi = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.K = K
        

    def forward(self, x_hidden, concept_matrix, market_value):
        device = torch.device(torch.get_device(x_hidden))

        # Predefined Concept Module
       
        market_value_matrix = market_value.reshape(market_value.shape[0], 1).repeat(1, concept_matrix.shape[1])
        stock_to_concept = concept_matrix * market_value_matrix
        
        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(stock_to_concept.shape[0], 1)
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)

        stock_to_concept_sum = stock_to_concept_sum + (torch.ones(stock_to_concept.shape[0], stock_to_concept.shape[1]).to(device))
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        hidden = torch.t(stock_to_concept).mm(x_hidden)
        
        hidden = hidden[hidden.sum(1)!=0]
        stock_to_concept = x_hidden.mm(torch.t(hidden))
        stock_to_concept = self.softmax_s2t(stock_to_concept)
        hidden = torch.t(stock_to_concept).mm(x_hidden)
        
        concept_to_stock = cal_cos_similarity(x_hidden, hidden) 
        concept_to_stock = self.softmax_t2s(concept_to_stock)

        p_shared_info = concept_to_stock.mm(hidden)
        p_shared_info = self.fc_ps(p_shared_info)

        p_shared_back = self.fc_ps_back(p_shared_info)
        output_ps = self.fc_ps_fore(p_shared_info)
        output_ps = self.leaky_relu(output_ps)

        pred_ps = self.fc_out_ps(output_ps).squeeze()
        
        # Hidden Concept Module
        h_shared_info = x_hidden - p_shared_back
        hidden = h_shared_info
        h_stock_to_concept = cal_cos_similarity(h_shared_info, hidden)

        dim = h_stock_to_concept.shape[0]
        diag = h_stock_to_concept.diagonal(0)
        h_stock_to_concept = h_stock_to_concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)

        row = torch.linspace(0, dim-1, dim).reshape([-1, 1]).repeat(1, self.K).reshape(1, -1).long().to(device)
        column = torch.topk(h_stock_to_concept, self.K, dim = 1)[1].reshape(1, -1)
        mask = torch.zeros([h_stock_to_concept.shape[0], h_stock_to_concept.shape[1]], device = h_stock_to_concept.device)
        mask[row, column] = 1
        h_stock_to_concept = h_stock_to_concept * mask
        h_stock_to_concept = h_stock_to_concept + torch.diag_embed((h_stock_to_concept.sum(0)!=0).float()*diag)
        hidden = torch.t(h_shared_info).mm(h_stock_to_concept).t()
        hidden = hidden[hidden.sum(1)!=0]

        h_concept_to_stock = cal_cos_similarity(h_shared_info, hidden)
        h_concept_to_stock = self.softmax_t2s(h_concept_to_stock)
        h_shared_info = h_concept_to_stock.mm(hidden)
        h_shared_info = self.fc_hs(h_shared_info)

        h_shared_back = self.fc_hs_back(h_shared_info)
        output_hs = self.fc_hs_fore(h_shared_info)
        output_hs = self.leaky_relu(output_hs)
        pred_hs = self.fc_out_hs(output_hs).squeeze()

        # Individual Information Module
        individual_info  = x_hidden - p_shared_back - h_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)
        pred_indi = self.fc_out_indi(output_indi).squeeze()
        # Stock Trend Prediction
        all_info = output_ps + output_hs + output_indi
        pred_all = self.fc_out(all_info).squeeze()
        return pred_all
