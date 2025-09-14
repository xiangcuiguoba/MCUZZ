#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run this file with : python afl-nn.py ./readelf -a

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
from collections import Counter
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MAX_FILE_SIZE = 10000
# MAX_BITMAP_SIZE = 2000
MAX_BITMAP_SIZE = 2123
round_cnt = 0
# Choose a seed for random initilzation
# seed = int(time.time())
seed = 12
np.random.seed(seed)
random.seed(seed)
set_random_seed(seed)
seed_list = glob.glob('./seeds/*')
new_seeds = glob.glob('./seeds/id_*')
SPLIT_RATIO = len(seed_list)
# get binary argv
argvv = sys.argv[1:]
Cluster_num = 8
data_dim = 3
Seed_list_cluster = []
Seed_list_axis = []
seeds_labels_axis = []
Select_seed_list = []
token_list = []
FOP = []


# process training data from afl raw data
def process_data():
    global MAX_BITMAP_SIZE
    global MAX_FILE_SIZE
    global SPLIT_RATIO
    global seed_list
    global new_seeds

    # shuffle training samples
    seed_list = glob.glob('./seeds/*')
    seed_list.sort()
    SPLIT_RATIO = len(seed_list)
    rand_index = np.arange(SPLIT_RATIO)
    np.random.shuffle(seed_list)
    new_seeds = glob.glob('./seeds/id_*')

    call = subprocess.check_output

    # get MAX_FILE_SIZE
    cwd = os.getcwd()
    max_file_name = call(['ls', '-S', cwd + '/seeds/']).decode('utf8').split('\n')[0].rstrip('\n')
    MAX_FILE_SIZE = os.path.getsize(cwd + '/seeds/' + max_file_name)

    # # create directories to save label, spliced seeds, variant length seeds, crashes and mutated seeds.
    os.path.isdir("./bitmaps/") or os.makedirs("./bitmaps")
    # os.path.isdir("./splice_seeds/") or os.makedirs("./splice_seeds")
    # os.path.isdir("./vari_seeds/") or os.makedirs("./vari_seeds")
    # os.path.isdir("./crashes/") or os.makedirs("./crashes")

    # obtain raw bitmaps
    raw_bitmap = {}
    tmp_cnt = []
    out = ''
    for f in seed_list:
        tmp_list = []
        try:
            # append "-o tmp_file" to strip's arguments to avoid tampering tested binary.
            if argvv[0] == './strip':
                out = call(['../../afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f] + ['-o', 'tmp_file'])
            else:
                out = call(['../../afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f])
        except subprocess.CalledProcessError:
            print("find a crash")
        for line in out.splitlines():
            edge = line.split(b':')[0]
            tmp_cnt.append(edge)
            tmp_list.append(edge)
        raw_bitmap[f] = tmp_list
    counter = Counter(tmp_cnt).most_common()

    # save bitmaps to individual numpy label
    label = [int(f[0]) for f in counter]
    bitmap = np.zeros((len(seed_list), len(label)))
    for idx, i in enumerate(seed_list):
        tmp = raw_bitmap[i]
        for j in tmp:
            if int(j) in label:
                bitmap[idx][label.index((int(j)))] = 1

    # label dimension reduction
    fit_bitmap = np.unique(bitmap, axis=1)
    print("data dimension" + str(fit_bitmap.shape))

    # save training data
    MAX_BITMAP_SIZE = fit_bitmap.shape[1]
    for idx, i in enumerate(seed_list):
        file_name = "./bitmaps/" + i.split('/')[-1]
        np.save(file_name, fit_bitmap[idx])


def Kmeans_train_generate():
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


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


def Compute_Distance_And_chose(Cluster_label, Cluster_Centroids_aixs) :  # 第一个参数是当前类的编号， 第二个参数是当前类的中心坐标
    # 计算当簇的所有种子距离中心坐标的距离
    distance_list = []

    for i in range(len(Seed_list_cluster[Cluster_label])):  # 变量当前簇中所有种子，对每个种子都计算一个距离
        if token_list[Cluster_label][i] == 0 :  # 如果遍历到的种子对应下标在token_list的值不为1，表示该种子被选择过了，跳过
            continue
        now_seed_name = Seed_list_cluster[Cluster_label][i]
        now_seed_axis = Seed_list_axis[Cluster_label][i]
        Distance = euclidean_distance(Cluster_Centroids_aixs, now_seed_axis)
        distance_list.append([now_seed_name, Distance])

    distance_list = sorted(distance_list, key=lambda x: x[1], reverse=True)

    if len(distance_list) == 0 :
        return -1
    else:
        index = Seed_list_cluster[Cluster_label].index(distance_list[0][0])
        token_list[Cluster_label][index] = 0    # 选择该种子，将token_list中相应标志为置为0
        return distance_list[0][0]              # 返回距离当前簇中心最远的种子名称


def check_sum_token(n_cluster):
    sum_token = 0
    for i in range(n_cluster):        # 如果token_list所有元素加合为0，表示所有种子都选过了
        sum_token += sum(token_list[i])
    if sum_token == 0 :
        return 0
    else:
        return 1


def Select_seed(Centroids, n_cluster) :
    # 我们将种子分为8类， 下次选择种子时，选择不是本簇的，并且离所在簇中心最远的种子
    global Select_seed_list
    # first_element = random.choice(seed_list)  # 随机挑选第一个种子
    # Select_seed_list.append(first_element)
    k = 1
    while check_sum_token(n_cluster):
        for i in range(n_cluster):
            cluster_centroids_axis = Centroids[i]   # 要挑选的第一个种子所属的簇的中心坐标
            if sum(token_list[i]) == 0 :   # 如果当前簇中，所有种子都被选择，跳过当前簇
                continue
            choose_seed = Compute_Distance_And_chose(i, cluster_centroids_axis)
            if choose_seed == -1:
                continue
            Select_seed_list.append([k, choose_seed, i, FOP[i]])
            k += 1
            # 将选中的种子，以 [序号, 种子名称, 所属类别, 类别占比因子] 的形式存放在列表Select_seed_list 中



def kmeans_multi_init(X, n_clusters, n_init=10, max_iter=300):
    """
    可以采用以下步骤来进行多次随机初始化聚类中心并取平均的操作：
        定义一个变量n，表示进行随机初始化聚类中心的次数。
        对于每次随机初始化，都使用随机种子来保证每次随机生成的初始聚类中心都不同。可以使用random.seed()函数来设置随机种子。
        对于每次随机初始化得到的聚类中心，都进行一次K-means聚类，并计算相应的聚类评价指标（如SSE、轮廓系数等）。
        将每次聚类得到的聚类中心坐标进行累加。
        当进行完所有的随机初始化后，将累加得到的聚类中心坐标除以n，得到最终的平均聚类中心坐标。
        使用最终的平均聚类中心坐标进行K-means聚类。
        通过这种方法，可以有效减小随机因素的影响，提高聚类的准确性和可靠性。

    随机初始化多次取平均的KMeans聚类算法

    参数：
        X: ndarray，数据集
        n_clusters: int，聚类中心数目
        n_init: int，初始化次数，默认为10
        max_iter: int，最大迭代次数，默认为300

    返回值：
        centroids: ndarray，聚类中心
        labels: ndarray，聚类结果标签

    在函数内部，我们通过对KMeans算法的n_init次随机初始化进行迭代，每次计算聚类损失，保存最小损失对应的聚类中心和标签作为最终结果。
    由于多次初始化是一个随机过程，因此通过这种方式可以减少聚类结果受随机因素影响的程度，得到更加鲁棒的聚类结果。
    """
    best_loss = np.inf
    for i in range(n_init):
        kmeans = KMeans(n_clusters=n_clusters, init='random', max_iter=max_iter)
        kmeans.fit(X)
        if kmeans.inertia_ < best_loss:
            best_loss = kmeans.inertia_
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_
    return centroids, labels


def KMeans_n(n_cluster, dim):
    global Seed_list_cluster
    global Seed_list_axis
    global seeds_labels_axis
    global Select_seed_list
    global token_list
    global FOP

    seeds, X = Kmeans_train_generate()
    # print(seeds)
    scaler = StandardScaler()
    data = scaler.fit_transform(X)
    pca = PCA(n_components=dim)
    data_reduced = pca.fit_transform(data)

    # kmeans聚类
    Centroids, labels = kmeans_multi_init(data_reduced, n_cluster)
    # 获取聚类的每个簇的中心坐标 与 每条数据的标签

    data_with_labels = [(seeds[i], labels[i]) for i in range(len(X))]       # 每一项元素都是 （种子名 ： 标签） 的形式
    label_with_axis = [(data_reduced[i], labels[i]) for i in range(len(X))]  # 每一项元素都是 (标签 ： 坐标) 的形式
    seeds_labels_axis = [[seeds[i], labels[i], data_reduced[i]] for i in range(len(X))]  # 每一项元素都是 （种子名 ： 标签 ： 坐标）的形式

    counter = Counter(labels).most_common()
    print('Clustering results1111111111:' + str(counter))

    data_dict = {}
    for i in range(n_cluster):
        data_dict[i] = []
    for data, class_label in data_with_labels:
        data_dict[class_label].append(data)
    Seed_list_cluster = [data_dict[i] for i in range(n_cluster)]  # 内有Cluster_num个数组，第i个数组存放的是第i簇的种子按照顺序的名称

    label_axis_dict = {}
    for i in range(n_cluster):
        label_axis_dict[i] = []
    for axis, the_label in label_with_axis:
        label_axis_dict[the_label].append(axis)
    Seed_list_axis = [label_axis_dict[i] for i in range(n_cluster)]  # 内有Cluster_num个数组，第i个数组存放的是第i簇的种子按照顺序的坐标


    for i in range(n_cluster):
        token = [1] * len(Seed_list_cluster[i])
        token_list.append(token)
    # Seed_list_cluster 与 Seed_list_axis 中每个元素都是一一对应的，对于种子A的名在Seed_list_cluster的坐标为Seed_list_cluster[i][j]
    # 则种子A的坐标在Seed_list_axis对应的位置为Seed_list_axis[i][j]
    # token_list中的每个元素标志其对应的种子是否已经被挑选，比如对于种子A的名在Seed_list_cluster的坐标为Seed_list_cluster[i][j]，
    # 如果token_list[i][j] 对应的元素为1，表示种子A还没有被挑选，若为0，则表示已经被挑选

    # 计算每个类的数据的占比因子
    for i in range(n_cluster):
        FOP.append(int(len(data_with_labels) / len(Seed_list_cluster[i])))
    print(FOP)
    FOP_set = set(FOP)
    if len(FOP_set) != len(FOP) :
        idx_lst = []
        for i, x in enumerate(FOP) :
            if FOP.count(x) > 1 :
                idx_lst.append(i)
        for i in range(len(idx_lst)-1) :
            FOP[idx_lst[i + 1]] += 1

    FOP_sum = 0
    for i in range(n_cluster):
        FOP_sum += FOP[i]*len(Seed_list_cluster[i])
    avg_FOP = FOP_sum / len(seed_list)

    print('-----------Select--------------')
    Select_seed(Centroids, n_cluster)

    with open('Select_list_info', 'w') as f:
        for i in range(len(Select_seed_list)) :
            f.write(str(Select_seed_list[i][0]) + ' ' + str(Select_seed_list[i][1].split('/')[-1]) + ' ' + str(Select_seed_list[i][2]) + ' ' + str(Select_seed_list[i][3]) + ' ' + str(avg_FOP) + "\n")
        # 将排序后的种子以 [序号, 种子名称, 所属类别, 类别占比因子] 的形形式写入文件
        f.close()

def gen_grad():
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
    KMeans_n(Cluster_num, data_dim)
    round_cnt = round_cnt + 1
    print(time.time() - t0)

gen_grad()