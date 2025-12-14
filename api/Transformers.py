"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import BertPreTrainedModel
# from transformers import AutoModel, AutoTokenizer
from .optimizer import *
from .kmeans_api import *
import numpy as np

GLOBAL_OPEN = True
KMEANS_OPEN = True
NORMAL_OPEN = False
KMEANS_BERT_OPEN = False
class SCCLBert(nn.Module):
    def __init__(self, bert_model, tokenizer, kmeans = None, kmeans_bert = None, cluster_centers=None, alpha=1.0):
        super(SCCLBert, self).__init__()
        
        self.tokenizer = tokenizer
        self.bert = bert_model
        self.emb_size = self.bert.config.hidden_size

        self.alpha = alpha
        self.kmeans = kmeans
        self.kmeans_bert = kmeans_bert
        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))
        
        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)
        print("sccl")
    
    def forward(self, input_ids, attention_mask, task_type="virtual"):
        if task_type == "evaluate":
            # print(input_ids)
            # print(attention_mask)
            return self.get_mean_embeddings(input_ids, attention_mask)
        
        elif task_type == "virtual":
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            # print("input:{}".format(input_ids_1.shape))

            # mean_output_1 = self.get_mean_embeddings(input_ids_1.cuda().float(),None)
            # mean_output_2 = self.get_mean_embeddings(input_ids_1.cuda().float(),None)
            return mean_output_1, mean_output_2
        
        elif task_type == "explicit":
            input_ids_1, input_ids_2, input_ids_3 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2, attention_mask_3 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_3 = self.get_mean_embeddings(input_ids_3, attention_mask_3)
            return mean_output_1, mean_output_2, mean_output_3
        
        else:
            raise Exception("TRANSFORMER ENCODING TYPE ERROR! OPTIONS: [EVALUATE, VIRTUAL, EXPLICIT]")

    def forward_by_kmeans(self, embed, is_bert=False):
        if is_bert:
            embed = embed.cpu().detach().numpy()
        else:
            embed = embed
        idx = self.kmeans.predict(embed)
        vec = torch.zeros(1,5)
        vec[0, idx] = 1
        return vec 
    
    def forward_by_kmeans_bert(self, embed, attn_mask):

        return self.kmeans_bert(embed.cpu().detach().numpy(), attn_mask)

    def get_mean_embeddings(self, input_ids, attention_mask):
        # print(input_ids)
        # print(attention_mask)
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        # mean_output = self.bert(input_ids)
        return mean_output
    

    def get_cluster_prob(self, embeddings):
        # embeddings = embeddings.cpu()
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2
    
    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1

def get_uniform(seq_s, time_s):
        # 以0.5s为间隔，采样和插值出带宽
        # print(time_s[0, -1])
        target_time = time_s[0, -1] - 20 * 1e3
        if target_time < 0:
            return torch.zeros(1,40), False
        # print(np.shape(time_s))
        input_seq = torch.zeros((1,40))
        idx = 0
        seq_idx = 0
        while True:
            
            if np.abs(target_time - time_s[0, -1]) < 0.5 * 1e3:
                break
            # print(idx)
            while idx < 40 and seq_idx < 40 and (target_time + 0.5 * 1e3 < time_s[0, idx]):
                input_seq[0, seq_idx] = seq_s[0, idx]
                target_time += 0.5 * 1e3
                seq_idx += 1
            
            while idx>=40 and seq_idx < 40 : 
                input_seq[0, seq_idx] = seq_s[0, -1]
                target_time += 0.5 * 1e3
                seq_idx += 1
            idx += 1
            # idx = np.min(idx,39)

        return input_seq, True

def roll_state(seq_s, seq_t):
    seq_s = np.roll(seq_s, -1, axis=1)
    seq_t = np.roll(seq_t, -1, axis=1)

def update_state(seq_s, seq_t, state, time_stamp, sccl):
    seq_s[0, -1] = state[0, -1] * 8 # 注意
    seq_t[0, -1] = time_stamp
    tmp = None
    input_seq, flag = get_uniform(seq_s, seq_t)
    if flag:
        attention_mask = torch.ones_like(input_seq).cuda().long()
        embeddings = sccl(input_seq.cuda().long(), attention_mask, "evaluate")
        if KMEANS_OPEN:
            tmp = sccl.forward_by_kmeans(input_seq, False)
        elif KMEANS_BERT_OPEN:
            tmp = sccl.forward_by_kmeans(embeddings, True)
        else:
            prob = sccl.get_cluster_prob(embeddings)
            tmp = prob
    else:
        tmp = torch.zeros(1, 5)
    
    return tmp, input_seq

def set_param(args):
    global GLOBAL_OPEN, KMEANS_OPEN, NORMAL_OPEN, KMEANS_BERT_OPEN
    GLOBAL_OPEN = args.global_open
    KMEANS_OPEN = args.kmeans_open
    NORMAL_OPEN = args.normal_open
    KMEANS_BERT_OPEN = args.kmeans_bert_open
    print(f"GLOBAL_OPEN {GLOBAL_OPEN} | KMEANS_OPEN {KMEANS_OPEN} | NORMAL_OPEN {NORMAL_OPEN} | NORMAL_OPEN {KMEANS_BERT_OPEN}\n")

def get_param():
    global GLOBAL_OPEN, KMEANS_OPEN, NORMAL_OPEN, KMEANS_BERT_OPEN
    print(f"GLOBAL_OPEN {GLOBAL_OPEN} | KMEANS_OPEN {KMEANS_OPEN} | NORMAL_OPEN {NORMAL_OPEN} | NORMAL_OPEN {KMEANS_BERT_OPEN}\n")
    return {
        "GLOBAL_OPEN" : GLOBAL_OPEN,
        "KMEANS_OPEN" : KMEANS_OPEN,
        "NORMAL_OPEN" : NORMAL_OPEN,
        "KMEANS_BERT_OPEN" : KMEANS_BERT_OPEN
    }

def load_model():
    print(f"GLOBAL_OPEN {GLOBAL_OPEN} | KMEANS_OPEN {KMEANS_OPEN} | KMEANS_OPEN {NORMAL_OPEN} | NORMAL_OPEN {KMEANS_BERT_OPEN}\n")
    bert, tokenizer = get_bert(None)
    cluster_centers = np.load("/home/zhongziyu/workspace/sccl-main-modified/cluster_centers.npy")
    kmeans = None
    kmeans_bert = None
    
    if KMEANS_OPEN: # 普通的kmeans
        print("USE KMEANS")
        kmeans = load_kmeans_with_centers()
    
    if KMEANS_BERT_OPEN: # 用bert的kmeans
        print("load_kmeans_bert")
        kmeans = load_kmeans_bert_with_centers()
    
    sccl = SCCLBert(bert, None, kmeans=kmeans, kmeans_bert=kmeans_bert, cluster_centers=cluster_centers, alpha=1.0).cuda()
    
    if not kmeans:
        if NORMAL_OPEN: # 无增强的sccl
            sccl.load_state_dict(torch.load("/home/public/zhongziyu/sccl-main-modified/results_new/SCCL.distilbert.SBERT.explicit.train_data.text.lr1e-05.lrscale100.SCCL.eta10.0.tmp0.5.alpha1.0.seed0/model.pth")["model"])
            print("normal no enhanced sccl")
        else: # 增强的sccl
            sccl.load_state_dict(torch.load("/home/public/zhongziyu/sccl-main-modified/results/SCCL.distilbert.SBERT.explicit.train_data.text.lr1e-05.lrscale100.SCCL.eta10.0.tmp0.5.alpha1.0.seed0/model.pth", map_location=torch.device('cuda'))["model"])
    
    print("finish_load_model")
    return sccl.eval()
