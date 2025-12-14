import torch
import numpy as np
from sklearn.cluster import KMeans
import joblib


# todo: 这里的KMeansBertModel是一个占位符，实际使用时需要替换为具体的BERT模型类
class KMeansBertModel:

    def __init__(self, bert, kmeans):
        self.bert = bert
        self.kmeans = kmeans

    def forward(self, input_ids, attention_mask):        # 获取BERT的输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 计算平均池化
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        prob = self.kmeans.predict(mean_output)
        return prob
    
def get_mean_embeddings(bert, input_ids, attention_mask):
    print("bert")
    bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
    attention_mask = attention_mask.unsqueeze(-1)
    mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    return mean_output

# corpus_embeddings = get_mean_embeddings(enc, text.cuda().long(), torch.ones_like(text).cuda().long())

def load_kmeans_with_centers():
    # 初始化 KMeans 模型，设置初始聚类中心
    print("load kmeans")
    kmeans = joblib.load('/home/public/zhongziyu/sccl-main-modified/kmeans_model.pkl')
    return kmeans

def load_kmeans_bert_with_centers():
    # 初始化 KMeans 模型，设置初始聚类中心
    print("load kmeans bert")
    kmeans = joblib.load('/home/public/zhongziyu/sccl-main-modified/kmeans_bert_model.pkl')
    return kmeans