import torch,sys
import numpy as np
import torch.nn as nn
sys.path.append(".")
from Config import DEVICE


class LSTM_CRF(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_tags):
        super(LSTM_CRF, self).__init__()
        self.num_tags=num_tags

        self.embed=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,bidirectional=True,batch_first=True)
        self.linear=nn.Linear(2*hidden_size,num_tags)

        self.EPS = 1e-8
        self.trans_matrix=nn.Parameter(torch.rand(num_tags,num_tags))

    def forward(self,X,X_len):
        X_embed=self.embed(X)

        X_embed_packed=nn.utils.rnn.pack_padded_sequence(X_embed,X_len,batch_first=True,enforce_sorted=False)
        X_embed_packed_hid,_=self.lstm(X_embed_packed)
        X_embed_hid,_=nn.utils.rnn.pad_packed_sequence(X_embed_packed_hid,batch_first=True)

        output=self.linear(X_embed_hid)

        return output


    def compute_loss(self,batch_sent,batch_tags,batch_length,mask):
        tag_score=self.forward(batch_sent,batch_length)

        emission_score=torch.sum(tag_score.gather(dim=2, index=batch_tags.unsqueeze(2)).squeeze(2)*mask,dim=1)

        trans_score=torch.zeros(size=(batch_tags.shape[0],)).to(DEVICE)
        if batch_tags.shape[1]>1:
            windows=batch_tags.unfold(dimension=1,size=2,step=1).reshape(-1,2)
            idx1,idx2=windows[:,0],windows[:,1]
            trans_score=(self.trans_matrix[idx1].gather(dim=1,index=idx2.reshape(-1,1)).squeeze(1).reshape(-1,batch_tags.shape[1]-1)*mask[:,1:]).sum(dim=1)

        sents_score=emission_score+trans_score

        log_norm_score=self.forward_algo_exp(tag_score,mask)

        log_prob=sents_score-log_norm_score

        return -log_prob

    def forward_algo_exp(self,tags_score,mask):
        start=tags_score[:,0,:].unsqueeze(2)
        for i in range(1,tags_score.shape[1]):
            M=self.trans_matrix+tags_score[:,i,:].unsqueeze(1).expand(-1,self.num_tags,-1)
            mask_i=mask[:,i].reshape(-1,1)
            start=(torch.logsumexp((start.expand(-1,-1,self.num_tags)+M),dim=1)*mask_i+start.squeeze(2)*(1-mask_i)).unsqueeze(2)
        return torch.logsumexp(start.squeeze(2),dim=1)

    def log(self,x):
        return torch.log(x+self.EPS)

    def predict(self,X):
        output=self.linear(self.lstm(self.embed(X))[0]).reshape(-1,self.num_tags).cpu()
        trans=self.trans_matrix.cpu().detach().numpy()

        LSTM_pred_prob=self.log(torch.softmax(output, dim=-1)).numpy()

        distance=LSTM_pred_prob[0,:]
        max_idx=[]

        for i in range(1,LSTM_pred_prob.shape[0]):
            max_idx_temp=[]
            distance_temp=np.zeros((self.num_tags,))
            for j in range(self.num_tags):
                temp=distance+trans[:,j]+LSTM_pred_prob[i][j]
                distance_temp[j]=np.max(temp)
                max_idx_temp.append(np.argmax(temp))

            max_idx.append(max_idx_temp)
            distance=distance_temp

        idx=[np.argmax(distance)]
        for i in range(len(max_idx)-1,-1,-1):
            idx.append(max_idx[i][idx[-1]])
        idx.reverse()

        return idx