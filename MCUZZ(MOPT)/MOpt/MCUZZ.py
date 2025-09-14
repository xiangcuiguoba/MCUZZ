#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run this file with : python MCUZZ.py ./readelf -a

import os
import sys
import glob
import math
import time
import keras
import random
import socket
import subprocess
import numpy as np
import tensorflow as tf
import keras.backend as K
import time  # 导入 time 模块
from collections import Counter
# from tensorflow import set_random_seed
from tensorflow.random import set_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from sklearn.cluster import OPTICS  # 引入 OPTICS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HOST = '127.0.0.1'
PORT = 12012

MAX_FILE_SIZE = 10000
MAX_BITMAP_SIZE = 2123
round_cnt = 0
seed = 12
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
seed_list = glob.glob('./seeds/*')
new_seeds = glob.glob('./seeds/id_*')
SPLIT_RATIO = len(seed_list)
# get binary argv
argvv = sys.argv[1:]
data_dim = 3
Seed_list_cluster = []
Seed_list_axis = []
seeds_labels_axis = []
Select_seed_list = []
token_list = []
FOP = []
path_length = {}


# 处理 afl 原始数据以生成训练数据
def process_data():
    global MAX_BITMAP_SIZE
    global MAX_FILE_SIZE
    global SPLIT_RATIO
    global seed_list
    global new_seeds
    global path_length

    # shuffle training samples
    # seed_list = glob.glob('./seeds/*')
    seed_list = glob.glob('./seeds/queue/*')
    seed_list.sort()
    SPLIT_RATIO = len(seed_list)
    rand_index = np.arange(SPLIT_RATIO)
    np.random.shuffle(seed_list)
    # new_seeds = glob.glob('./seeds/id_*')
    new_seeds = glob.glob('./seeds/queue/id_*')

    call = subprocess.check_output

    # 获取 MAX_FILE_SIZE
    cwd = os.getcwd()
    max_file_name = call(['ls', '-S', cwd + '/seeds/queue/']).decode('utf8').split('\n')[0].rstrip('\n')
    MAX_FILE_SIZE = os.path.getsize(cwd + '/seeds/queue/' + max_file_name)

    # 创建保存标签、位图等的目录
    os.path.isdir("./bitmaps/") or os.makedirs("./bitmaps")

    # 获取原始位图
    raw_bitmap = {}
    tmp_cnt = []
    out = ''
    for f in seed_list:
        tmp_list = []
        try:
            if argvv[0] == './strip':
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f] + ['-o', 'tmp_file'])
            else:
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f])
        except subprocess.CalledProcessError:
            print("find a crash")
        for line in out.splitlines():
            edge = line.split(b':')[0]
            tmp_cnt.append(edge)
            tmp_list.append(edge)
        raw_bitmap[f] = tmp_list
    counter = Counter(tmp_cnt).most_common()

    # 将位图保存到单独的 numpy 标签
    label = [int(f[0]) for f in counter]
    bitmap = np.zeros((len(seed_list), len(label)))
    for idx, i in enumerate(seed_list):
        path_count = 0
        tmp = raw_bitmap[i]
        for j in tmp:
            if int(j) in label:
                bitmap[idx][label.index((int(j)))] = 1
                path_count += 1
        path_length[i] = path_count
    # label dimension reduction
    fit_bitmap = np.unique(bitmap, axis=1)
    print("data dimension" + str(fit_bitmap.shape))

    # 保存训练数据
    MAX_BITMAP_SIZE = fit_bitmap.shape[1]
    for idx, i in enumerate(seed_list):
        file_name = "./bitmaps/" + i.split('/')[-1]
        np.save(file_name, fit_bitmap[idx])

# 种子文件的列表和它们对应的位图数据矩阵
def optics_train_generate():
    global seed_list
    bitmap = np.zeros((len(seed_list), MAX_BITMAP_SIZE))
    np.random.shuffle(seed_list)
        # load a batch of training data
    for i in range(len(seed_list)):
        file_name = "./bitmaps/" + seed_list[i].split('/')[-1] + ".npy"
        # file_name = "./bitmaps/" + seed_list[i].split('\\')[-1] + ".npy"
        # 获取文件名称，不需要路径，与./bitmaps 组成种子执行覆盖文件的路径
        bitmap[i] = np.load(file_name)

    return seed_list, bitmap


def Compute_rarity(cluster_label,now_seed_name, now_seed_FOP, avg_FOP, avg_path_len, ori_labels):
    now_seed_path_len = path_length[now_seed_name]
    # 获取当前种子的簇标签
    if avg_FOP == 0 or avg_path_len == 0:
        print(f"Warning: avg_FOP or avg_path_len is zero. avg_FOP: {avg_FOP}, avg_path_len: {avg_path_len}")
        return 0  # 避免除以零错误，返回默认值

    # 非噪声簇（cluster_label != -1）
    if ori_labels != -2:
        if cluster_label <= int(ori_labels):
            # 使用原公式
            score = (now_seed_FOP / avg_FOP) * (1 + 0.75 * (now_seed_path_len / avg_path_len))
        else:
            # 噪声簇（cluster_label == -1），使用调整后的公式
            score = (now_seed_FOP / avg_FOP) * (1 + 0.75 * (now_seed_path_len / avg_path_len))* (1 + 0.75)
    else:
        score = (now_seed_FOP / avg_FOP) * (1 + 0.75 * (now_seed_path_len / avg_path_len))
    return score


def Compute_Rarity_Score_And_chose(Cluster_label, Cluster_FOP, avg_FOP, avg_path_len, ori_labels) :
    # 计算当簇的所有种子距离中心坐标的距离
    distance_list = []

    for i in range(len(Seed_list_cluster[Cluster_label])):  # 变量当前簇中所有种子，对每个种子都计算一个距离
        if token_list[Cluster_label][i] == 0 :  # 如果遍历到的种子对应下标在token_list的值不为1，表示该种子被选择过了，跳过
            continue
        now_seed_name = Seed_list_cluster[Cluster_label][i]
        now_seed_FOP = Cluster_FOP
        rarity_score = Compute_rarity(i,now_seed_name, now_seed_FOP, avg_FOP, avg_path_len, ori_labels)
        distance_list.append([now_seed_name, rarity_score])

    distance_list = sorted(distance_list, key=lambda x: x[1], reverse=True)

    if len(distance_list) == 0 :
        return -1
    else:
        index = Seed_list_cluster[Cluster_label].index(distance_list[0][0])
        token_list[Cluster_label][index] = 0    # 选择该种子，将token_list中相应标志为置为0
        return distance_list[0]              # 返回距离当前簇中心最远的种子名称

def check_sum_token(n_cluster):
    sum_token = 0
    for i in range(n_cluster):        # 如果token_list所有元素加合为0，表示所有种子都选过了
        sum_token += sum(token_list[i])
    if sum_token == 0 :
        return 0
    else:
        return 1


def Select_seed(Centroids, n_cluster, avg_FOP, avg_path_len,ori_labels) :
    # 我们将种子分为n类， 下次选择种子时，选择不是本簇的，并且离所在簇中心最远的种子
    global Select_seed_list
    k = 1
    while check_sum_token(n_cluster):
        for i in range(n_cluster):
            cluster_FOP = FOP[i]   # 要挑选的第一个种子所属的簇的中心坐标
            if sum(token_list[i]) == 0 :   # 如果当前簇中，所有种子都被选择，跳过当前簇
                continue
            choose_seed = Compute_Rarity_Score_And_chose(i, cluster_FOP, avg_FOP, avg_path_len, ori_labels)
            if choose_seed == -1:
                continue
            Select_seed_list.append([k, choose_seed[0], i, choose_seed[1]])
            k += 1
            # 将选中的种子，以 [序号, 种子名称, 所属类别, 类别占比因子] 的形式存放在列表Select_seed_list 中


def optics_multi_init(X, min_samples=50, xi=0.02, min_cluster_size=0.02):
    # 根据实际样本数调整 min_samples
    min_samples = min(min_samples, len(X))  # 确保 min_samples 不会超过样本数
    X = np.array(X)
    # 创建 OPTICS 实例
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)

    # 拟合数据并获取标签
    labels = optics.fit_predict(X)
    print(f"data  split into {len(np.unique(labels))} clusters.")
    return labels



def handle_noise_cluster(seeds, labels, data_reduced,  ori_labels,  n_clusters=6):
    """
    对 -1 簇进行进一步处理，如果种子数量超过 threshold，则进行 KMeans 聚类，
    并将新簇标记为连续编号，同时尽量保证簇大小均匀。
    """
    noise_cluster_indices = [i for i, label in enumerate(labels) if label == -1]


    # 输出原始簇大小
    original_cluster_sizes = {label: sum(labels == label) for label in set(labels)}
    print(f"Original cluster sizes: {original_cluster_sizes}")
    total_seeds = len(seeds)
    # 如果噪声簇中的种子数超过 threshold
    if len(noise_cluster_indices) > total_seeds/3:
        print(f"Processing noise cluster with {len(noise_cluster_indices)} seeds.")

        # 提取 -1 簇的数据
        noise_seeds = [seeds[i] for i in noise_cluster_indices]
        noise_data = data_reduced[noise_cluster_indices]

        # 使用 KMeans 对 -1 簇再进行一次聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, init='k-means++')
        new_labels = kmeans.fit_predict(noise_data)

        # 获取每个新簇的大小
        cluster_sizes = [sum(new_labels == i) for i in range(n_clusters)]
        print(f"Cluster sizes after KMeans: {cluster_sizes}")

        # 获取当前标签的最大值，为新簇分配连续编号
        max_label = max(labels) + 1
        ori_labels =  max(labels)
        new_cluster_labels = list(range(max_label, max_label + n_clusters))  # 新簇名字

        # 更新原始标签
        for i, idx in enumerate(noise_cluster_indices):
            labels[idx] = max_label + new_labels[i]

        # 输出分成的簇的名字和每簇的大小
        final_cluster_sizes = {lbl: sum(labels == lbl) for lbl in new_cluster_labels}
        print(f"Noise cluster split into {n_clusters} sub-clusters.")
        print(f"old max cluster labels : {ori_labels}")
        print(f"New cluster labels: {new_cluster_labels}")
        print(f"Final cluster sizes: {final_cluster_sizes}")

    return seeds, labels, ori_labels



def optics_nn(n_cluster, dim):
    global Seed_list_cluster
    global Seed_list_axis
    global seeds_labels_axis
    global Select_seed_list
    global token_list
    global FOP
    global ori_labels
    seeds, X = optics_train_generate()
    scaler = StandardScaler()
    data = scaler.fit_transform(X)
    pca = PCA(n_components=data_dim)
    data_reduced = pca.fit_transform(data)

    # 使用 OPTICS 聚类
    labels = optics_multi_init(data_reduced)

    # 处理 -1 簇
    ori_labels = -2  # 初始化变量，防止未定义错误
    seeds, labels, ori_labels= handle_noise_cluster(seeds, labels, data_reduced, ori_labels)
    print(f"data split into {len(np.unique(labels))} sub-----clusters.")

    data_with_labels = [(seeds[i], labels[i]) for i in range(len(X))]  # 每一项元素都是（种子名 ： 标签）的形式
    label_with_axis = [(data_reduced[i], labels[i]) for i in range(len(X))]  # 每一项元素都是(标签 ： 坐标)的形式
    seeds_labels_axis = [[seeds[i], labels[i], data_reduced[i]] for i in range(len(X))]  # 每一项元素都是（种子名 ： 标签 ： 坐标）的形式

    counter = Counter(labels).most_common()
    print('Clustering results:' + str(counter))

    data_dict = {}
    for label in np.unique(labels):
        data_dict[label] = []
    for data, class_label in data_with_labels:
        data_dict[class_label].append(data)
    Seed_list_cluster = [data_dict[label] for label in np.unique(labels)]

    label_axis_dict = {}
    for label in np.unique(labels):
        label_axis_dict[label] = []
    for axis, the_label in label_with_axis:
        label_axis_dict[the_label].append(axis)
    Seed_list_axis = [label_axis_dict[label] for label in np.unique(labels)]

    for i in range(len(Seed_list_cluster)):
        token = [1] * len(Seed_list_cluster[i])
        token_list.append(token)

    # 计算每个类的数据的占比因子
    for i in range(len(Seed_list_cluster)):
        if len(Seed_list_cluster[i]) > 0:  # 避免空簇导致除零错误
            FOP.append(int(len(data_with_labels) / len(Seed_list_cluster[i])))
        else:
            FOP.append(0)  # 如果簇为空，将 FOP 设置为 0，防止除零
            print(f"Cluster {i} is empty.")  # 添加调试输出

    print(f"Initial FOP values: {FOP}")

    # 处理重复的 FOP 值
    FOP_set = set(FOP)
    if len(FOP_set) != len(FOP):
        idx_lst = []
        for i, x in enumerate(FOP):
            if FOP.count(x) > 1:
                idx_lst.append(i)
        for i in range(len(idx_lst) - 1):
            FOP[idx_lst[i + 1]] += 1

    print(f"FOP values after adjustment: {FOP}")

    # 计算 FOP_sum，避免空簇引起的错误
    FOP_sum = sum(FOP[i] * len(Seed_list_cluster[i]) for i in range(len(Seed_list_cluster)) if FOP[i] > 0)

    # 检查 seed_list 的长度是否大于 0，避免除零
    if len(seed_list) > 0:
        avg_FOP = FOP_sum / len(seed_list)
    else:
        print("Warning: seed_list is empty, setting avg_FOP to 1 as default.")
        avg_FOP = 1

    # 计算 avg_path_len
    path_length_sum = sum(path_length[i] for i in seed_list if i in path_length)
    if len(seed_list) > 0:
        avg_path_len = path_length_sum / len(seed_list)
    else:
        print("Warning: seed_list is empty, setting avg_path_len to 1 as default.")
        avg_path_len = 1

    print(f"Calculated avg_FOP: {avg_FOP}, avg_path_len: {avg_path_len}")

    print('-----------Select--------------')
    Select_seed(np.unique(labels), len(Seed_list_cluster),avg_FOP, avg_path_len, ori_labels)

    with open('Select_list_info', 'w') as f:
        for i in range(len(Select_seed_list)):
            f.write(str(Select_seed_list[i][0]) + ' ' + str(Select_seed_list[i][1].split('/')[-1]) + ' ' + str(Select_seed_list[i][2]) + ' ' + str(Select_seed_list[i][3]) + ' ' + str(avg_FOP) + "\n")


def gen_grad(data):
    global round_cnt
    global Seed_list_cluster
    global Seed_list_axis
    global seeds_labels_axis
    global Select_seed_list
    global token_list
    global FOP
    Seed_list_cluster = []
    Seed_list_axis = []
    seeds_labels_axis = []
    Select_seed_list = []
    token_list = []
    FOP = []
    t0 = time.time()
    process_data()
    optics_nn(len(Seed_list_cluster),data_dim)  # 运行 OPTICS 聚类
    round_cnt += 1
    print("运行时间: %f 秒" % (time.time() - t0))


# gen_grad(data)

def setup_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    print('sock binded ')
    sock.listen(1)
    conn, addr = sock.accept()
    print('connected by neuzz execution moduel12012 ' + str(addr))
    # gen_grad(b"train")
    # print('pyfirst round train successed ')
    # conn.sendall(b"start")
    # print('py first sendstart')
    while True:
        data = conn.recv(1024)
        if not data:
            break
        else:
            print('pynext round train start')
            gen_grad(data)
            print('pynext round train successed')
            conn.sendall(b"start")
            print('py sendsended')

    conn.close()


if __name__ == '__main__':
    setup_server()


