import torch
from torch  import nn
import math
import numpy as np
from sklearn import metrics
from scipy.interpolate import RegularGridInterpolator
from random import randint
import random
import os
from hyptorch import hynn,pmath
from pytorch_metric_learning import miners, losses
import torch.nn.functional as F
from operator import truediv
from models import create_graph
import matplotlib.pyplot as plt
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def compute_loss(network_output: torch.Tensor, train_samples_gt_onehot: torch.Tensor, train_label_mask: torch.Tensor):
    real_labels = train_samples_gt_onehot
    we = -torch.mul(real_labels,torch.log(network_output))
    we = torch.mul(we, train_label_mask)
    pool_cross_entropy = torch.sum(we)

    return pool_cross_entropy

def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx = (train_samples_gt!=0).float()        # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()       # 有效标签的个数
        correct_prediction = torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
        OA= correct_prediction.cpu() / available_label_count
        return OA

def one_hot(train_hi_gt):
    num, = np.array(train_hi_gt.shape)
    ont_hot_label = [] 
    for i in range(num):
        temp = np.zeros(3, dtype=np.int64)
        if train_hi_gt[i] != 0:
            temp[int(train_hi_gt[i]) - 1] = 1
        ont_hot_label.append(temp)
    ont_hot_label = np.reshape(ont_hot_label, [num, 3])
    return ont_hot_label

def compute_hiloss(y_res,train_hi_gt,train_index,class_num,x,y,z):
    hiloss = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    for i in range (class_num):
        loss = criterion(y_res[train_index,i,:], train_hi_gt[:,i])
        hiloss = hiloss+loss
    return hiloss
def label_to_one_hot(data_gt, class_num):

    height, = data_gt.shape
    ont_hot_label = [] 
    for i in range(height):
        temp = np.zeros(class_num, dtype=np.int64)
        if data_gt[i] != 0:
            temp[int(data_gt[i]) - 1] = 1
        ont_hot_label.append(temp)
    ont_hot_label = np.reshape(ont_hot_label, [height, class_num])
    return ont_hot_label
def compute_hiloss_new(y_res,train_hi_gt,train_index,class_num,h):
    hiloss = 0
    train_hi_gt = train_hi_gt.cpu()
    a, = train_index.shape
    for i in range (class_num):
        res = torch.softmax(y_res[train_index,i,:], 1)
        real_labels = label_to_one_hot(train_hi_gt[:,i]+1, 3)
        real_labels = torch.from_numpy(real_labels.astype(np.float32)).to(device)
        we = -torch.mul(real_labels,torch.log(res))
        index_obj = np.array(np.where(train_hi_gt[:,i]==2))
        para = torch.ones((a,3)).to(device)
        para[index_obj,:] = h
        we = torch.mul(para,we)
        loss = torch.sum(we)

        hiloss = hiloss+loss
    return hiloss

def init_grid(n_spixels_expc, w, h):
    # n_spixels >= n_spixels_expc
    nw_spixels = math.ceil(math.sqrt(w*n_spixels_expc/h))
    nh_spixels = math.ceil(math.sqrt(h*n_spixels_expc/w))

    n_spixels = nw_spixels*nh_spixels   # Actual number of spixels

    if n_spixels > w*h:
        raise ValueError("Superpixels must be fewer than pixels!")
        
    w_spixel, h_spixel = (w+nw_spixels-1) // nw_spixels, (h+nh_spixels-1) // nh_spixels
    rw, rh = w_spixel*nw_spixels-w, h_spixel*nh_spixels-h

    if (rh/2 + h_spixel) < 0 or (rw/2 + w_spixel) < 0 or (rh/2-h_spixel) > 0 or (rw/2-w_spixel) > 0:
        raise ValueError("The expected number of superpixels does not fit the image size!")

    y = np.array([-1, *np.arange((h_spixel-1)/2, h+rh, h_spixel), h+rh])-rh/2
    x = np.array([-1, *np.arange((w_spixel-1)/2, w+rw, w_spixel), w+rw])-rw/2

    s = np.arange(n_spixels).reshape(nh_spixels, nw_spixels).astype(np.int32)
    s = np.pad(s, ((1,1),(1,1)), 'edge')
    f = RegularGridInterpolator((y, x), s, method='nearest')

    pts = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pts = np.stack(pts, axis=-1)
    init_idx_map = f(pts).astype(np.int32)
    
    return init_idx_map, n_spixels, nw_spixels, nh_spixels

class FeatureConverter:
    def __init__(self, eta_pos=2, gamma_clr=0.1):
        super().__init__()
        self.eta_pos = eta_pos
        self.gamma_clr = gamma_clr

    def __call__(self, feats, nw_spixels, nh_spixels):
        # Do not require grad
        b, c, h, w = feats.size()

        pos_scale = self.eta_pos*max(nw_spixels/w, nh_spixels/h)   
        coords = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device)), 0)
        coords = coords[None].repeat(feats.shape[0], 1, 1, 1).float()
        # print(pos_scale)
        feats = torch.cat([feats, pos_scale*coords], 1)#(1,202,145,145)
        # feats.requires_grad = True
        return feats
    
def setup_seed(seed):

    #seed=randint(1,5000)
    #seed=1
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False


def evaluate_performance_all(network_output,train_samples_gt,train_samples_gt_onehot, m, n, class_count, Test_GT, require_AA_KPP=False,printFlag=True):
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    zeros = torch.zeros([m * n]).to(device).float()
    if False==require_AA_KPP:
        with torch.no_grad():
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count
            
            return OA
    else:
        with torch.no_grad():
            #计算OA
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count
            OA=OA.cpu().numpy()
            
            # 计算AA
            zero_vector = np.zeros([class_count])
            output_data=network_output.cpu().numpy()
            train_samples_gt=train_samples_gt.cpu().numpy()
            train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
            
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            for z in range(output_data.shape[0]):
                if ~(zero_vector == output_data[z]).all():
                    idx[z] += 1
            
            count_perclass = np.zeros([class_count])
            correct_perclass = np.zeros([class_count])
            for x in range(len(train_samples_gt)):
                if train_samples_gt[x] != 0:
                    count_perclass[int(train_samples_gt[x] - 1)] += 1
                    if train_samples_gt[x] == idx[x]:
                        correct_perclass[int(train_samples_gt[x] - 1)] += 1
            test_AC_list = correct_perclass / count_perclass
            test_AA = np.average(test_AC_list)

            # 计算KPP
            test_pre_label_list = []
            test_real_label_list = []
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            idx = np.reshape(idx, [m, n])
            for ii in range(m):
                for jj in range(n):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(idx[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                              test_real_label_list.astype(np.int16))
            test_kpp = kappa

            # 输出
            if printFlag:
                print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                print('acc per class:')
                print(test_AC_list)

            OA_ALL.append(OA)
            AA_ALL.append(test_AA)
            KPP_ALL.append(test_kpp)
            AVG_ALL.append(test_AC_list)
           
            return OA,OA_ALL,AA_ALL,KPP_ALL,AVG_ALL
        
        
def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
def ConfusionMatrix(output,label,index,class_num):
    conf_matrix = np.zeros((class_num+1,class_num+1)).astype(int)
    preds = torch.argmax(output, 1)
    preds=np.array(preds.cpu()).astype(int)
    preds = preds
    label=np.array(label.cpu()).astype(int)
    label = label
    for p, t in zip(preds[index], label[index]):
        conf_matrix[p, t] += 1
    return conf_matrix


def find_topnode(edge_list,class_num):
    edge_array = np.array(edge_list)
    size = len(edge_list)
    sub_edge_array = edge_array[class_num:size,:].reshape(-1)
    counts = np.bincount(sub_edge_array)
    most_common_value = np.argmax(counts) 
    return   most_common_value

def bolicDistance_triloss(embeddings, labels,c,margin):
    miner = miners.TripletMarginMiner(margin, type_of_triplets = 'all')
    hard_pairs = miner(embeddings, labels)
    anchor_idx = hard_pairs[0]
    positive_idx = hard_pairs[1]
    negative_idx = hard_pairs[2]
    dis = hynn.HyperbolicDistanceLayer(c = 0.5)
    size,b=embeddings.shape
    embeddings1 =embeddings.repeat_interleave(size, dim=0)
    embeddings2 =embeddings.repeat(size, 1)
    D = dis(embeddings1, embeddings2,c).reshape(size,size)
    ap_dists = D[anchor_idx,positive_idx]
    an_dists = D[anchor_idx,negative_idx]
    loss = F.relu(ap_dists - an_dists + margin)
    loss = torch.mean(loss)
    return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'all')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
def eucDistance_triloss(embbing,label,margin):
    dis = TripletLoss(margin = margin)
    loss = dis(embbing,label)
    return loss
def  interclass_dis(embbing,label,c,class_num):
    number,chanl = embbing.shape
    mean_embbing =  torch.zeros(class_num,chanl)
    for i in range (class_num):
        index = torch.where(label == i+1)
        mean_embbing[i,:] = torch.mean(embbing[index],0)
    dis = hynn.HyperbolicDistanceLayer(c = c)
    D = torch.zeros((class_num,class_num))
    for i in range(class_num):
        repeat_mean_embbing = mean_embbing[i].reshape(-1,1)
        repeat_mean_embbing = repeat_mean_embbing.repeat(1,class_num).permute([1, 0])
        D[i,:] = dis(repeat_mean_embbing,mean_embbing,c).reshape(-1)
    return D

def  interclass_dis_euc(embbing,label,c,class_num):
    number,chanl = embbing.shape
    mean_embbing =  torch.zeros(class_num,chanl)
    for i in range (class_num):
        index = torch.where(label == i+1)
        mean_embbing[i,:] = torch.mean(embbing[index],0)
    D = torch.zeros((class_num,class_num))
    D = torch.cdist(mean_embbing,mean_embbing)
    return D
    
def output_combine(Tree_stu, class_num, i):

    # 当前节点的孩子
    child_nodes = np.where(Tree_stu[i] == 1)[0]

    branch_number = child_nodes.size

    # [分支数, 真实类别数]
    marix = torch.zeros(branch_number, class_num)

    # DFS扫描位置
    scan_row = i

    # ==================================
    # 提取某行中的真实叶子类别
    # ==================================
    def get_leaf_classes(row):

        child_nodes = np.where(Tree_stu[row] == 1)[0]

        leaf_classes = child_nodes[
            np.where(child_nodes < class_num)
        ]

        return leaf_classes

    # ==================================
    # root节点
    # ==================================
    if i == 0:

        scan_row += 1

        for branch_idx in range(branch_number):

            while scan_row < Tree_stu.shape[0]:

                leaf_classes = get_leaf_classes(scan_row)

                marix[branch_idx, leaf_classes] = 1

                current_children = np.where(
                    Tree_stu[scan_row] == 1
                )[0]

                scan_row += 1

                # 当前节点没有粗类别孩子
                # 当前子树结束
                if np.sum(current_children >= class_num) == 0:
                    break

    # ==================================
    # 一个叶子 + 一个粗类别
    # ==================================
    elif np.sum(child_nodes < class_num) == 1:

        # 第一个叶子分支
        leaf_classes = get_leaf_classes(scan_row)

        marix[0, leaf_classes] = 1

        scan_row += 1

        # 第二个粗类别分支
        while scan_row < Tree_stu.shape[0]:

            leaf_classes = get_leaf_classes(scan_row)

            marix[1, leaf_classes] = 1

            current_children = np.where(
                Tree_stu[scan_row] == 1
            )[0]

            scan_row += 1

            if np.sum(current_children >= class_num) == 0:
                break

    # ==================================
    # 两个叶子节点
    # ==================================
    elif np.sum(child_nodes < class_num) == 2:

        leaf_classes = get_leaf_classes(scan_row)

        marix[0, leaf_classes[0]] = 1
        marix[1, leaf_classes[1]] = 1

    return marix

def output_combine_old(Tree_stu,class_num,i):
    find_branch = np.where(Tree_stu[i]==1)[0]
    branch_number = find_branch.size
    marix = torch.zeros(branch_number,class_num)
    w = i
    if i ==0:   
        w = w+1
        for num in range (branch_number):
            for wqe in range (100):
                find_node = np.where(Tree_stu[w]==1)[0]
                find_rel_class =  find_node[np.where(find_node<class_num)]
                for o in range (find_rel_class.size):
                    marix[num,find_rel_class[o]]=1
                w= w+1
                if find_node[np.where(find_node>class_num)].size ==0:
                    break
    if find_branch[np.where(find_branch<class_num)].size ==1 :
            find_node = np.where(Tree_stu[w]==1)[0]
            find_rel_class =  find_node[np.where(find_node<class_num)]
            for o in range (find_rel_class.size):
                marix[0,find_rel_class[o]]=1
            w = w+1
            for wqe in range (100):
                find_node = np.where(Tree_stu[w]==1)[0]
                find_rel_class =  find_node[np.where(find_node<class_num)]
                for o in range (find_rel_class.size):
                    marix[1,find_rel_class[o]]=1
                w= w+1
                if find_node[np.where(find_node>class_num)].size ==0:
                    break
    if find_branch[np.where(find_branch<class_num)].size ==2 :
        find_node = np.where(Tree_stu[w]==1)[0]
        find_rel_class =  find_node[np.where(find_node<class_num)]
        marix[0,find_rel_class[0]]=1
        marix[1,find_rel_class[1]]=1

    return marix

def output_combine_leafnode(Tree_stu,class_num,i):
    find_branch = np.where(Tree_stu[i]==1)[0]
    branch_number = find_branch.size
    marix = torch.zeros(branch_number,class_num)
    w = i
    if i ==0:   
        w = w+1
        for num in range (branch_number):
            for wqe in range (100):
                find_node = np.where(Tree_stu[w]==1)[0]
                find_rel_class =  find_node[np.where(find_node<class_num)]
                for o in range (find_rel_class.size):
                    marix[num,find_rel_class[o]]=1
                w= w+1
                if find_node[np.where(find_node>class_num)].size ==0:
                    break
    if find_branch[np.where(find_branch<class_num)].size ==1 :
            find_node = np.where(Tree_stu[w]==1)[0]
            find_rel_class =  find_node[np.where(find_node<class_num)]
            for o in range (find_rel_class.size):
                marix[0,find_rel_class[o]]=1
            w = w+1
            for wqe in range (100):
                find_node = np.where(Tree_stu[w]==1)[0]
                find_rel_class =  find_node[np.where(find_node<class_num)]
                for o in range (find_rel_class.size):
                    marix[1,find_rel_class[o]]=1
                w= w+1
                if find_node[np.where(find_node>class_num)].size ==0:
                    break
    if find_branch[np.where(find_branch<class_num)].size ==2 :
        find_node = np.where(Tree_stu[w]==1)[0]
        find_rel_class =  find_node[np.where(find_node<class_num)]
        marix[0,find_rel_class[0]]=1
        marix[1,find_rel_class[1]]=1

    return marix

def output_convert(input,marix,device):
    num,a = marix.shape
    N,c = input.shape
    output = torch.zeros(N,num).to(device)
    for i in range(num):
        index = np.where(marix[i]==1)[0]
        index_num = index.size
        for w in range (index_num):
            output[:,i] = output[:,i]+input[:,index[w]]
    return output

def output_convert_leafnode(input,marix,device):
    num, = marix.shape
    N,c = input.shape
    output = torch.zeros(N,num).to(device)
    for i in range(num):
        output[:,i] = input[:,marix[i]]
    return output
def imshow(data, class_num, name):
    colormap = np.zeros((20, 3))
    colormap[1, :] = [128/255, 128/255, 128/255]
    colormap[2, :] = [0, 255/255, 0]
    colormap[3, :] = [0, 255/255, 255/255]
    colormap[4, :] = [0, 128/255, 0]
    colormap[5, :] = [255/255, 0, 255/255]
    colormap[6, :] = [255/255, 255/255, 0]
    colormap[7, :] = [0, 0, 128/255]
    colormap[8, :] = [255/255, 0, 0]
    colormap[9, :] = [128/255, 0, 0]
    colormap[10, :] = [0, 0, 255/255]
    colormap[11, :] = [237/255, 145/255, 33/255]
    colormap[12, :] = [221/255, 160/255, 221/255]
    colormap[13, :] = [156/255, 102/255, 31/255]
    colormap[14, :] = [125/255, 38/255, 205/255]
    colormap[15, :] = [51/255, 161/255, 201/255]
    colormap[16, :] = [255/255, 127/255, 80/255]
    colormap[17, :] = [128/255, 51/255, 255/255]
    colormap[18, :] = [33/255, 128/255, 51/255]
    colormap[19, :] = [112/255, 130/255, 255/255]
    colormap[20, :] = [237/255, 127/255, 80/255]
    colormap[21, :] = [128/255, 237/255, 255/255]
    colormap[22, :] = [255/255, 51/255, 128/255]
    colormap =colormap/255
    h, w = data.shape
    truthmap = np.zeros((h, w, 3), dtype=np.float32)

    # **向量化映射类别到颜色**
    valid_mask = (data > 0) & (data <= class_num)
    truthmap[valid_mask] = colormap[data[valid_mask] - 1]  # 类别从1开始，索引从0开始调整

    # 归一化到 0-255
    truthmap = (truthmap*255).astype(np.uint8)

    # 保存图片
    image = Image.fromarray(truthmap)
    image.save(name + '.png')

    # **在服务器上使用 Matplotlib 展示图片**
    plt.figure(figsize=(8, 8))
    plt.imshow(truthmap)
    plt.axis('off')  # 隐藏坐标轴
    plt.title(name)
    plt.show()  