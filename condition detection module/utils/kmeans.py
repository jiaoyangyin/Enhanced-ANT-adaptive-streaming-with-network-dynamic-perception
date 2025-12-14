"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import joblib

def get_mean_embeddings(bert, input_ids, attention_mask):
        print("bert")
        bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    
def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return token_feat


def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    for i, batch in enumerate(train_loader):

        text, label = batch['text'], batch['label']
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features)
        
        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings.detach().numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().numpy()), axis=0)
        print(all_embeddings.shape)
    # Perform KMeans clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))
    print(pred_labels.shape)
    print(true_labels.shape)

    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    
    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(), clustering_model.cluster_centers_.shape))
    return clustering_model.cluster_centers_


def get_kmeans_centers_new(enc, train_loader, test_loader, num_classes, max_length):
    # enc = None
    for i, batch in enumerate(train_loader):

        text, label = batch['text'], batch['label']
        # text_new, label_new = batch['text'], batch['label'][160000:]
        if enc != None:
            corpus_embeddings = get_mean_embeddings(enc, text.cuda().long(), torch.ones_like(text).cuda().long())
        else:
            corpus_embeddings = text
        print(corpus_embeddings.shape)
        
        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings.cpu().detach().numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.cpu().detach().numpy()), axis=0)

    # Perform KMeans clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    true_labels = all_labels.reshape(-1)
    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))
    d = np.where(pred_labels == true_labels)
    print(np.shape(d))
    joblib.dump(clustering_model, 'kmeans_model.pkl')

    # FMI = fowlkes_mallows_score(all_embeddings, pred_labels)
    # CONTOUR = silhouette_score(all_embeddings, pred_labels)
    # CALINSKI = calinski_harabasz_score(all_embeddings, pred_labels)
    # print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(), clustering_model.cluster_centers_.shape))
    # print("FMI:{:.3f}, CONTOUR:{:.3f}, CALINSKI:{:.3f}".format(0,CONTOUR,CALINSKI))

    # #TSNE可视化
    # tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(all_embeddings)
    # # tsne 归一化， 这一步可做可不做
    # x_min, x_max = tsne.min(0), tsne.max(0)
    # tsne_norm = (tsne - x_min) / (x_max - x_min)
    # idxs_array = []
    # plt.figure()
    # for i in range(5):
    #     idxs = np.where(pred_labels == i)
    #     tsne_data = tsne_norm[idxs]
    #     plt.scatter(tsne_data[:, 0], tsne_data[:, 1], 1, label='class {}'.format(i))
    # plt.legend(loc='upper left')
    # plt.savefig("res.png")
    # exit()



    return clustering_model.cluster_centers_

