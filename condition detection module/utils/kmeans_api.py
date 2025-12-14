import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_mean_embeddings(bert, input_ids, attention_mask):
    print("bert")
    bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
    attention_mask = attention_mask.unsqueeze(-1)
    mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    return mean_output

# corpus_embeddings = get_mean_embeddings(enc, text.cuda().long(), torch.ones_like(text).cuda().long())

def load_kmeans_with_centers():
    # 初始化 KMeans 模型，设置初始聚类中心
    clustering_centers = np.load("/home/zhongziyu/workspace/sccl-main-modified/cluster_centers_new.npy")
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, init=clustering_centers, n_init=1, max_iter=1)
    return kmeans
