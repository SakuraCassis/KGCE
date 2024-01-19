import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_normal_



def cal_matrix(A):
    D = A.sum(axis=1)
    D = torch.tensor(D, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)
    mask = D > 0
    D_masked = torch.where(mask, D, torch.tensor(1.0))
    normalized_A = torch.where(mask[:, None], A / D_masked[:, None], A)
    return normalized_A

#tcp layer
def xavier_init(m):
    if type(m) == nn.Linear:
        xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

    
    
class KGE(nn.Module):
    def __init__(self, data, args): #, n_entity, n_relation 
        super(KGE, self).__init__()

        self.embed_dim = args.dim
        self.features = {}  # initial embedding
        self.final_emb = {} 
        self.node_num = 0
        self.cls = data['feature']['C'].shape[0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.lr_emb = args.lr_emb
        self.l2 = args.l2
        self.lambda1 = args.lambda1


        for nti in data['feature']:
            self.features[nti] = torch.tensor(data['feature'][nti]).float().to(self.device)
            self.node_num += len(data['feature'][nti])

        self.HTS = data['HTS'] 
        self.adjs = []
        for si in data['adjs']:
            adjs_si = []
            for adj in si:
                adjs_si.append(cal_matrix(adj).to(self.device)) 
                # adjs_si.append(adj)
            self.adjs.append(adjs_si)

        self.cls_adjs={}
        htsi=data['HTS'][1]   
        for a,nti in enumerate(htsi):
            adjsi = data['adjs'][1][a]
            for i in range(1,len(htsi)-a-1): 
                adjsi = np.matmul(data['adjs'][1][a+i], adjsi)
            label=cal_matrix(adjsi.T) 
            if a == 3:
                break
            self.cls_adjs[nti]=label

        self.linear1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, 1)
        self.linear3 = nn.Linear(self.embed_dim, 1)
        self.linears = nn.ModuleList([nn.ModuleList([nn.Linear(self.embed_dim, self.cls) for _ in range(len(data['HTS'][0]))]), nn.ModuleList([nn.Linear(self.embed_dim, self.cls) for _ in range(len(data['HTS'][1]))])])  
        self.linears.apply(xavier_init)

        self.TCPConfidenceLayer = LinearLayer(128, 1)
        self.grus = nn.ModuleList([nn.GRUCell(self.embed_dim, self.embed_dim) for _ in range(len(data['HTS']))])
        self.grus.apply(xavier_init)

        self.final_rel_emb = nn.Parameter(torch.randn(12, 128))   

    def forward(self, u_si, u_di):

        # embedding
        nti_emb = {} 
        
        self.loss_tcp = 0 # tcp_loss
        self.l2_loss_emb = 0

        ## self aggregation
        for nti in self.features:  
            nti_emb[nti] = []
            self_emb = self.linear1(self.features[nti])
            nti_emb[nti].append(self_emb)
           

        ##  Aggregator
        for i, (htsi, adjs) in enumerate(zip(self.HTS, self.adjs)):  
            h_hat=[]
            h=[]
            for a,nti in enumerate(htsi):

                if a==0:
                    h_hat.append(self.features[nti]) 
                    label= self.cls_adjs[nti].to(self.device)
                    pred =F.softmax(self.linears[i][a](h_hat[a]),dim=1)
                    tcp=torch.bmm(pred.reshape(-1,1,self.cls),label.reshape(-1,self.cls,1))
                    self.loss_tcp += nn.CrossEntropyLoss()(pred,label)
                    h.append(torch.mm(adjs[a], tcp.squeeze(1)*h_hat[a])) 

                else:
                    state = self.grus[i](self.features[nti], h[a-1])
                    h_hat.append(state)
                    nti_emb[nti].append(state) 
                    if a==len(htsi)-1:
                        break

                    label= self.cls_adjs[nti].to(self.device)
                    pred =F.softmax(self.linears[i][a](h_hat[a]),dim=1) 
                    tcp=torch.bmm(pred.reshape(-1,1,self.cls),label.reshape(-1,self.cls,1))
                    self.loss_tcp += nn.CrossEntropyLoss()(pred,label)
                    h.append(torch.mm(adjs[a], tcp.squeeze(1)*h_hat[a])) 


        # B. Neighborhood Information Integrating
        for nti in nti_emb:  
            embs = nti_emb[nti]
            # self attention 
            a1 = self.linear2(embs[0]) # self_emb  
            a2 = self.linear3(embs[1]) # hts_emb  
            a=torch.cat([a1,a2], dim=0) 
            alpha = F.softmax(F.leaky_relu(a,0).reshape(2,len(embs[0]),1),dim=1) 
            embs=torch.cat([embs[0],embs[1]], dim=0).reshape(2,len(embs[0]),128)
            emb_i = F.relu(torch.sum(alpha * embs, dim=0))
            self.final_emb[nti] = emb_i

        self.final_ent_emb = torch.cat([self.final_emb[nti] for nti in ['P','V','C','A']], dim=0) # final entity emb #16092x128 ## 所有数据集的pkl中 node都改为了 pvca，entity编码的顺序也是pvca

        # C. Optimization via Relational Metric learning

        self.u_s = torch.tensor(u_si) #shape=[self.sample_num,3] #node pair with same types
        self.u_d = torch.tensor(u_di) #shape=[self.sample_num,4] #node pair with distinct types

        # print("self.u_d",self.u_d.shape)

        self.u_i_d = self.u_d[:,0]
        self.u_j_d = self.u_d[:,1]
        self.label_d = self.u_d[:,2].to(self.device)
        self.r = self.u_d[:,3]

        self.u_i_s = self.u_s[:,0]
        self.u_j_s = self.u_s[:,1]
        self.label_s = self.u_s[:,2].to(self.device)

        self.u_i_embedding_d = self.final_ent_emb[self.u_i_d]
        self.u_j_embedding_d = self.final_ent_emb[self.u_j_d]
        self.u_i_embedding_s = self.final_ent_emb[self.u_i_s]
        self.u_j_embedding_s = self.final_ent_emb[self.u_j_s]

        M_r = self.final_rel_emb[self.r]
        self.inner_product_d = torch.sum(M_r * F.tanh(self.u_i_embedding_d + self.u_j_embedding_d), axis=1)
        self.inner_product_s = torch.sum(self.u_i_embedding_s * self.u_j_embedding_s, axis=1)

        self.base_loss_emb = -torch.sum(torch.nn.LogSigmoid()(self.label_d * self.inner_product_d))\
        -torch.sum(torch.nn.LogSigmoid()(self.label_s * self.inner_product_s)) 

        self.tcp_loss = self.loss_tcp
        
        self.l2_loss = self.l2 * sum(torch.norm(param,2) #l2 norm
            for name,param in self.named_parameters() if 'bias' not in name)

        self.loss_emb = self.base_loss_emb + self.l2_loss_emb * self.l2 + self.lambda1*self.tcp_loss

        
        return self.loss_emb , self.final_ent_emb, self.final_rel_emb

    
class PARS(nn.Module):

    def __init__(self, args):
        super(PARS, self).__init__()

        self.lr_rs = args.lr_rs
        self.l2 = args.l2

        # AGRE 
        self.dim = args.dim
        self.p = args.p
        self.path_len = args.path_len 

        self.rnn = nn.RNN(2 * args.dim, 2 * args.dim)
        self.weight_predict = nn.Linear(2 * args.dim, 1)
        self.weight_path = nn.Parameter(torch.randn(args.p, args.p))
        self.weight_attention = nn.Linear(2 * args.dim, 1)
        
        self.entity_embedding_matrix =nn.Parameter(args.embeddings)
        self.relation_embedding_matrix = nn.Parameter(args.relation)



    def forward(self, paths_list, relation_dict):

        # AGRE
        embeddings_list = self.get_embedding(paths_list, relation_dict)
        embeddings = torch.cat(embeddings_list, dim=0)
        h = self.rnn(embeddings)[0][-1]
        h = h.reshape(-1, self.p, 2 * self.dim)
        h = torch.sigmoid(torch.matmul(self.weight_path, h))

        ## attention
        attention = torch.sigmoid(self.weight_attention(h))
        attention = torch.softmax(attention, dim=1)
        final_hidden_states = (attention * h).sum(dim=1)
        self.predicts = torch.sigmoid(self.weight_predict(final_hidden_states).reshape(-1))

        return self.predicts

    def get_embedding(self, paths_list, relation_dict):

        embeddings_list = []
        zeros = torch.zeros(self.p, self.dim)
        if torch.cuda.is_available():
            zeros = zeros.to(self.entity_embedding_matrix.data.device)

        for i in range(self.path_len+1):
            i_entity_embedding_list = []
            i_relation_embedding_list = []
            for paths in paths_list:

                if len(paths) == 0:
                    i_entity_embedding_list.append(zeros)
                    i_relation_embedding_list.append(zeros)
                    continue

                if i != self.path_len:
                    relation_embeddings = self.relation_embedding_matrix[[relation_dict[(path[i], path[i+1])] for path in paths]]
                else:
                    relation_embeddings = zeros
                i_relation_embedding_list.append(relation_embeddings)
                entity_embeddings = self.entity_embedding_matrix[[path[i] for path in paths]]
                i_entity_embedding_list.append(entity_embeddings)


            relations_embeddings = torch.cat(i_relation_embedding_list, dim=0)
            entities_embeddings = torch.cat(i_entity_embedding_list, dim=0)
            embeddings = torch.cat([entities_embeddings, relations_embeddings], dim=-1).reshape(1, -1, 2 * self.dim)
            embeddings_list.append(embeddings)
        return embeddings_list
