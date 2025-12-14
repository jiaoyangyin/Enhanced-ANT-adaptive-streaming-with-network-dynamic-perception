"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
from models.Transformers import SCCLBert
import dataloader.dataloader as dataloader
from training import SCCLvTrainer, BertvTrainer
from utils.kmeans import get_kmeans_centers_new, get_kmeans_centers, get_mean_embeddings
from utils.logger import setup_path, set_global_random_seed
from utils.optimizer import get_optimizer, get_bert, get_enc
import numpy as np
from utils.metric import Confusion
from sklearn import cluster
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
import pickle

def run(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    # dataset loader
    train_loader, test_loader = dataloader.explict_augmentation_loader_new(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader_new(args)
    # train_loader = dataloader.explict_augmentation_loader_new(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader_new(args)
    # model
    torch.cuda.set_device(args.gpuid[0])
    bert, tokenizer = get_bert(args)
    enc = get_enc()
    print("here")
    
    # initialize cluster centers
    
    # cluster_centers = get_kmeans_centers(bert, tokenizer, train_loader, args.num_classes, args.max_length)
    # cluster_centers = np.load("/home/zhongziyu/workspace/sccl-main-modified/cluster_centers_new.npy")
    
    cluster_centers = get_kmeans_centers_new(bert, train_loader, test_loader, args.num_classes, args.max_length)
    # cluster_centers = get_kmeans_centers_new(None, train_loader, test_loader, args.num_classes, args.max_length)
    # exit()

    # np.save("cluster_centers_new.npy",np.array(cluster_centers))
    # exit()

    print("cluster_centers:{}".format(cluster_centers))
    model = SCCLBert(bert, None, cluster_centers=cluster_centers, alpha=args.alpha) 
    model = model.cuda()
    print("here")
    # optimizer 
    optimizer = get_optimizer(model, args)
    print("here")
    
    trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader, args)
    # print("here")
    
    trainer.train()
    
    return None

def run_bert(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    # dataset loader
    train_loader, test_loader = dataloader.explict_augmentation_loader_new(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader_new(args)
    # train_loader = dataloader.explict_augmentation_loader_new(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader_new(args)
    # model
    torch.cuda.set_device(args.gpuid[0])
    bert, tokenizer = get_bert(args)
    enc = get_enc()
    print("bert train here")
    
    # initialize cluster centers
    
    # cluster_centers = get_kmeans_centers(bert, tokenizer, train_loader, args.num_classes, args.max_length)
    # cluster_centers = np.load("/home/zhongziyu/workspace/sccl-main-modified/cluster_centers_new.npy")
    
    cluster_centers = get_kmeans_centers_new(bert, train_loader, test_loader, args.num_classes, args.max_length)
    np.save("cluster_centers_new.npy",np.array(cluster_centers))

    print("cluster_centers:{}".format(cluster_centers))
    model = SCCLBert(bert, None, cluster_centers=cluster_centers, alpha=args.alpha) 
    model = model.cuda()
    print("here")
    # optimizer 
    optimizer = get_optimizer(model, args)
    print("here")
    
    trainer = BertvTrainer(model, tokenizer, optimizer, train_loader, args)
    
    trainer.train()
    
    return None

def test(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    # dataset loader
    train_loader, test_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader_new(args)
    # model
    torch.cuda.set_device(args.gpuid[0])
    bert, tokenizer = get_bert(args)
    
    # cluster_centers = get_kmeans_centers_new(bert, train_loader, test_loader, args.num_classes, args.max_length)
    cluster_centers = np.load("cluster_centers_new.npy")
    print("cluster_centers:{}".format(cluster_centers))
    # np.save("cluster_centers.npy",np.array(cluster_centers))
    model = SCCLBert(bert, None, cluster_centers=cluster_centers, alpha=args.alpha) 
    model = model.cuda()
    model.load_state_dict(torch.load("/home/public/zhongziyu/sccl-main-modified/results_0909/SCCL.distilbert.SBERT.explicit.train_data.text.lr1e-05.lrscale100.SCCL.eta10.0.tmp0.5.alpha1.0.seed0/model.pth", map_location='cuda:2')["model"])
    
        
    print('---- {} evaluation batches ----'.format(0))
    
    model.eval()
    for i, batch in enumerate(train_loader):
        with torch.no_grad():
            text, label = batch['text'], batch['label'] 
            # feat = self.get_batch_token(text)
            feat = model(text.cuda().long(), torch.ones_like(text).cuda().long(), task_type="evaluate")
            # embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")
            embeddings = feat.cuda()
            model_prob = model.get_cluster_prob(embeddings)
            if i == 0:
                all_labels = label
                all_embeddings = embeddings.detach()
                all_prob = model_prob
            else:
                all_labels = torch.cat((all_labels, label), dim=0)
                all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                all_prob = torch.cat((all_prob, model_prob), dim=0)
                
    # Initialize confusion matrices
    confusion, confusion_model = Confusion(args.num_classes), Confusion(args.num_classes)
    
    all_pred = all_prob.max(1)[1]
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(args.num_classes)
    acc_model = confusion_model.acc()

    kmeans = cluster.KMeans(n_clusters=args.num_classes, random_state=args.seed)
    embeddings = all_embeddings.cpu().numpy()
    kmeans.fit(embeddings)
    pred_labels = torch.tensor(kmeans.labels_.astype(int))
    
    # clustering accuracy 
    confusion.add(pred_labels, all_labels)
    # confusion.optimal_assignment(args.num_classes)
    acc = confusion.acc()

    ressave = {"acc":acc, "acc_model":acc_model}
    ressave.update(confusion.clusterscores())

    # 计算内部指标
    # FMI = fowlkes_mallows_score(embeddings, pred_labels)
    CONTOUR = silhouette_score(embeddings, pred_labels)
    CALINSKI = calinski_harabasz_score(embeddings, pred_labels)
    # FMI_MODEL = fowlkes_mallows_score(embeddings, all_pred)
    # print(all_pred.shape)
    CONTOUR_MODEL = silhouette_score(embeddings, all_pred.cpu())
    # CONTOUR_MODEL = -1
    CALINSKI_MODEL = calinski_harabasz_score(embeddings, all_pred.cpu())
    # CALINSKI_MODEL = -1

    print('[Representation] Clustering scores:',confusion.clusterscores()) 
    print('[Representation] ACC: {:.3f}'.format(acc)) 
    # print('[Representation] FMI: {:.3f}'.format(FMI)) 
    print('[Representation] CONTOUR: {:.3f}'.format(CONTOUR)) 
    print('[Representation] CALINSKI: {:.3f}'.format(CALINSKI)) 
    print('[Model] Clustering scores:',confusion_model.clusterscores()) 
    print('[Model] ACC: {:.3f}'.format(acc_model))
    # print('[Model] FMI: {:.3f}'.format(FMI_MODEL)) 
    print('[Model] CONTOUR: {:.3f}'.format(CONTOUR_MODEL)) 
    print('[Model] CALINSKI: {:.3f}'.format(CALINSKI_MODEL))
    # 作图 画出类别分布图
    
    #TSNE可视化
    tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(all_embeddings.cpu())
    # tsne 归一化， 这一步可做可不做
    x_min, x_max = tsne.min(0), tsne.max(0)
    tsne_norm = (tsne - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(5):
        idxs = np.where(all_pred.cpu() == i)
        tsne_data = tsne_norm[idxs]
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], 1, label='class {}'.format(i))
    plt.legend(loc='lower right')
    plt.savefig("./outputs/sccl.pdf")
    
    return None

def test_no_enhanced(args):
    print('test no enhanced')
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    # dataset loader
    train_loader, test_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader_new(args)
    # model
    torch.cuda.set_device(args.gpuid[0])
    bert, tokenizer = get_bert(args)
    
    # cluster_centers = get_kmeans_centers_new(bert, train_loader, test_loader, args.num_classes, args.max_length)
    cluster_centers = np.load("cluster_centers_new.npy")
    print("cluster_centers:{}".format(cluster_centers))
    # np.save("cluster_centers.npy",np.array(cluster_centers))
    model = SCCLBert(bert, None, cluster_centers=cluster_centers, alpha=args.alpha) 
    model = model.cuda()
    model.load_state_dict(torch.load("/home/public/zhongziyu/sccl-main-modified/results_new/SCCL.distilbert.SBERT.explicit.train_data.text.lr1e-05.lrscale100.SCCL.eta10.0.tmp0.5.alpha1.0.seed0/model.pth", map_location='cuda:2')["model"])
    
        
    print('---- {} evaluation batches ----'.format(0))
    
    model.eval()
    for i, batch in enumerate(train_loader):
        with torch.no_grad():
            text, label = batch['text'], batch['label'] 
            # feat = self.get_batch_token(text)
            feat = model(text.cuda().long(), torch.ones_like(text).cuda().long(), task_type="evaluate")
            # embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")
            embeddings = feat.cuda()
            model_prob = model.get_cluster_prob(embeddings)
            if i == 0:
                all_labels = label
                all_embeddings = embeddings.detach()
                all_prob = model_prob
            else:
                all_labels = torch.cat((all_labels, label), dim=0)
                all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                all_prob = torch.cat((all_prob, model_prob), dim=0)
                
    # Initialize confusion matrices
    confusion, confusion_model = Confusion(args.num_classes), Confusion(args.num_classes)
    
    all_pred = all_prob.max(1)[1]
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(args.num_classes)
    acc_model = confusion_model.acc()

    kmeans = cluster.KMeans(n_clusters=args.num_classes, random_state=args.seed)
    embeddings = all_embeddings.cpu().numpy()
    kmeans.fit(embeddings)
    pred_labels = torch.tensor(kmeans.labels_.astype(int))
    
    # clustering accuracy 
    confusion.add(pred_labels, all_labels)
    # confusion.optimal_assignment(args.num_classes)
    acc = confusion.acc()

    ressave = {"acc":acc, "acc_model":acc_model}
    ressave.update(confusion.clusterscores())

    # 计算内部指标
    # FMI = fowlkes_mallows_score(embeddings, pred_labels)
    CONTOUR = silhouette_score(embeddings, pred_labels)
    CALINSKI = calinski_harabasz_score(embeddings, pred_labels)
    # FMI_MODEL = fowlkes_mallows_score(embeddings, all_pred)
    # print(all_pred.shape)
    CONTOUR_MODEL = silhouette_score(embeddings, all_pred.cpu())
    # CONTOUR_MODEL = -1
    CALINSKI_MODEL = calinski_harabasz_score(embeddings, all_pred.cpu())
    # CALINSKI_MODEL = -1

    print('[Representation] Clustering scores:',confusion.clusterscores()) 
    print('[Representation] ACC: {:.3f}'.format(acc)) 
    # print('[Representation] FMI: {:.3f}'.format(FMI)) 
    print('[Representation] CONTOUR: {:.3f}'.format(CONTOUR)) 
    print('[Representation] CALINSKI: {:.3f}'.format(CALINSKI)) 
    print('[Model] Clustering scores:',confusion_model.clusterscores()) 
    print('[Model] ACC: {:.3f}'.format(acc_model))
    # print('[Model] FMI: {:.3f}'.format(FMI_MODEL)) 
    print('[Model] CONTOUR: {:.3f}'.format(CONTOUR_MODEL)) 
    print('[Model] CALINSKI: {:.3f}'.format(CALINSKI_MODEL))
    # 作图 画出类别分布图
    
    #TSNE可视化
    tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(all_embeddings.cpu())
    # tsne 归一化， 这一步可做可不做
    x_min, x_max = tsne.min(0), tsne.max(0)
    tsne_norm = (tsne - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(5):
        idxs = np.where(all_pred.cpu() == i)
        tsne_data = tsne_norm[idxs]
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], 1, label='class {}'.format(i))
    plt.legend(loc='lower right')
    plt.savefig("./outputs/sccl_no_enhanced.pdf")
    
    return None

def test_kmeans(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)
    print("kmeans")

    # dataset loader
    train_loader, test_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader_new(args)
    # model
    torch.cuda.set_device(args.gpuid[0])
    for i, batch in enumerate(train_loader):

        text, label = batch['text'], batch['label']
        corpus_embeddings = torch.Tensor(text)
        print(corpus_embeddings.shape)
        
        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings.cpu().detach().numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.cpu().detach().numpy()), axis=0)

    # Perform KMeans clustering
    confusion = Confusion(args.num_classes)
    clustering_model = KMeans(n_clusters=args.num_classes)
    # 删除包含任何NaN值的行
    # 找到所有不包含NaN的行的索引
    # non_nan_rows = ~np.isnan(all_embeddings).any(axis=1)

    # # 筛选出不包含NaN的行
    # all_embeddings = all_embeddings[non_nan_rows]
    # all_embeddings = np.nan_to_num(all_embeddings, nan=0)
    clustering_model.fit(all_embeddings)
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(clustering_model, f)
    cluster_assignment = clustering_model.labels_
    print(cluster_assignment.shape)

    true_labels = all_labels.reshape(-1)
    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))
    # d = np.where(pred_labels == true_labels)
    # print(np.shape(d))
    # FMI = fowlkes_mallows_score(all_embeddings, pred_labels)
    CONTOUR = silhouette_score(all_embeddings, pred_labels)
    CALINSKI = calinski_harabasz_score(all_embeddings, pred_labels)
    DBI = davies_bouldin_score(all_embeddings, pred_labels)
    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(), clustering_model.cluster_centers_.shape))
    print("FMI:{:.3f}, CONTOUR:{:.3f}, CALINSKI:{:.3f}, DBI:{:.3f}".format(0,CONTOUR,CALINSKI,DBI))

    #TSNE可视化
    tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(all_embeddings)
    # tsne 归一化， 这一步可做可不做
    x_min, x_max = tsne.min(0), tsne.max(0)
    tsne_norm = (tsne - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(5):
        idxs = np.where(pred_labels == i)
        tsne_data = tsne_norm[idxs]
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], 1, label='class {}'.format(i))
    plt.legend(loc='upper left')
    plt.savefig("./outputs/kmeans.pdf")

    return None


def test_bert(args):
# dataset loader
    set_global_random_seed(args.seed)
    train_loader, test_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader_new(args)
    bert, tokenizer = get_bert(args)
    print("kmeans bert")
    cluster_centers = np.load("cluster_centers_new.npy")
    print("cluster_centers:{}".format(cluster_centers))
    # np.save("cluster_centers.npy",np.array(cluster_centers))
    model = SCCLBert(bert, None, cluster_centers=cluster_centers, alpha=args.alpha) 
    model = model.cuda()
    model.load_state_dict(torch.load("/home/public/zhongziyu/sccl-main-modified/results_bert/SCCL.distilbert.SBERT.virtual.train_data.text.lr1e-05.lrscale100.SCCL.eta10.0.tmp0.5.alpha1.0.seed0/model_5.pth", map_location='cuda:2')["model"])
    # model
    # torch.cuda.set_device(args.gpuid[0])
    for i, batch in enumerate(train_loader):

        text, label = batch['text'], batch['label']
        # print(bert.device)
        # corpus_embeddings = get_mean_embeddings(bert, text.cuda().long(), torch.ones_like(text).cuda().long())
        # print(corpus_embeddings.shape)
        feat = model(text.cuda().long(), torch.ones_like(text).cuda().long(), task_type="evaluate")
        # embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")
        corpus_embeddings = feat.cuda()
        
        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings.cpu().detach().numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.cpu().detach().numpy()), axis=0)

    # Perform KMeans clustering
    confusion = Confusion(args.num_classes)
    clustering_model = KMeans(n_clusters=args.num_classes)
    clustering_model.fit(all_embeddings)
    # 保存权重
    with open('kmeans_bert_model.pkl', 'wb') as f:
        pickle.dump(clustering_model, f)

    cluster_assignment = clustering_model.labels_
    print(cluster_assignment.shape)

    true_labels = all_labels.reshape(-1)
    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))
    # d = np.where(pred_labels == true_labels)
    # print(np.shape(d))
    # FMI = fowlkes_mallows_score(all_embeddings, pred_labels)
    CONTOUR = silhouette_score(all_embeddings, pred_labels)
    CALINSKI = calinski_harabasz_score(all_embeddings, pred_labels)
    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(), clustering_model.cluster_centers_.shape))
    print("FMI:{:.3f}, CONTOUR:{:.3f}, CALINSKI:{:.3f}".format(0,CONTOUR,CALINSKI))

    #TSNE可视化
    tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(all_embeddings)
    # tsne 归一化， 这一步可做可不做
    x_min, x_max = tsne.min(0), tsne.max(0)
    tsne_norm = (tsne - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(5):
        idxs = np.where(pred_labels == i)
        tsne_data = tsne_norm[idxs]
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], 1, label='class {}'.format(i))
    plt.legend(loc='upper left')
    plt.savefig("./outputs/bert_kmeans.pdf")

    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local') 
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=100, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')
    
    parser.add_argument('--bert', type=str, default='distilroberta', help="")
    parser.add_argument('--use_pretrain', type=str, default='BERT', choices=["BERT", "SBERT", "PAIRSUPCON"])
    
    parser.add_argument('--mode', type=str, default="test")
    
    # Dataset
    parser.add_argument('--datapath', type=str, default='../data_segment_all/')
    parser.add_argument('--dataname', type=str, default='train_data', help="")
    parser.add_argument('--num_classes', type=int, default=4, help="")
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--augmentation_1', type=str, default='text1')
    parser.add_argument('--augmentation_2', type=str, default='text2')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=1000)
    # contrastive learning
    parser.add_argument('--objective', type=str, default='contrastive')
    parser.add_argument('--augtype', type=str, default='virtual', choices=['virtual', 'explicit'])
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=1, help="")
    
    # Clustering
    parser.add_argument('--alpha', type=float, default=1.0)
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args


if __name__ == '__main__':
    # import subprocess
       
    args = get_args(sys.argv[1:])

    # if args.train_instance == "sagemaker" and False:
    #     run(args)
    #     subprocess.run(["aws", "s3", "cp", "--recursive", args.resdir, args.s3_resdir])
    # else:
    if args.mode == "train":
        run(args)
    elif args.mode == 'train_bert':
        run_bert(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "test_no_enhanced":
        test_no_enhanced(args)
    elif args.mode == "kmeans":
        test_kmeans(args)
    elif args.mode == "bert":
        test_bert(args)
    else:
        print("left")
            



    
