import torch
from torch  import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,cohen_kappa_score
import time
from models import create_graph,MFS_HiNet,utils
import time
from random import randint,sample
import copy
from operator import truediv
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.notebook import tqdm
import os
import random
from skimage.segmentation import slic,felzenszwalb
from IPython import display
from thop import profile
from mmengine.optim import build_optim_wrapper
def train_node_classifier(
    device,
    max_epoch,
    learning_rate,
    path_weight,
    data,
    gt_local,
    gt_global,
    train_index,
    val_index,
    test_index,
    train_index_global,
    val_index_global,
    test_index_global,
    seed,
    load_model,
    parent_model_name,
    current_model_name,
    curvature,
    layer,
    hi_margin,
    euc_margin,
    inheritance_alpha,
    use_parameter_inheritance,
    kernel_size,
):

    height, width, bands = data.shape

    setup_seed(seed)

    # =========================================================
    # Local Label
    # =========================================================
    train_label, test_label, val_label = create_graph.get_label(
        gt_local,
        train_index,
        val_index,
        test_index
    )

    unique_numbers, train_label = label_change(train_label)
    unique_numbers, val_label = label_change(val_label)

    train_gt = train_label.reshape(height, width)
    test_gt = test_label.reshape(height, width)
    val_gt = val_label.reshape(height, width)

    class_num = int(np.max(train_gt))
    class_num_test = int(np.max(test_gt))

    train_onehot = create_graph.label_to_one_hot(train_gt, class_num)
    test_onehot = create_graph.label_to_one_hot(test_gt, class_num_test)
    val_onehot = create_graph.label_to_one_hot(val_gt, class_num)

    train_label = torch.from_numpy(train_label.astype(np.int64)).to(device)
    test_label = torch.from_numpy(test_label.astype(np.int64)).to(device)
    val_label = torch.from_numpy(val_label.astype(np.int64)).to(device)

    train_onehot = torch.from_numpy(train_onehot.astype(np.int64)).to(device)
    test_onehot = torch.from_numpy(test_onehot.astype(np.int64)).to(device)
    val_onehot = torch.from_numpy(val_onehot.astype(np.int64)).to(device)

    train_label_shift = train_label - 1
    val_label_shift = val_label - 1

    # =========================================================
    # Global Label
    # =========================================================
    train_label_global, test_label_global, val_label_global = create_graph.get_label(
        gt_global,
        train_index_global,
        val_index_global,
        test_index_global
    )

    train_gt_global = train_label_global.reshape(height, width)
    test_gt_global = test_label_global.reshape(height, width)
    val_gt_global = val_label_global.reshape(height, width)

    class_num_global = int(np.max(train_gt_global))

    train_onehot_global = create_graph.label_to_one_hot(
        train_gt_global,
        class_num_global
    )

    test_onehot_global = create_graph.label_to_one_hot(
        test_gt_global,
        class_num_global
    )

    val_onehot_global = create_graph.label_to_one_hot(
        val_gt_global,
        class_num_global
    )

    train_label_global = torch.from_numpy(
        train_label_global.astype(np.int64)
    ).to(device)

    test_label_global = torch.from_numpy(
        test_label_global.astype(np.int64)
    ).to(device)

    val_label_global = torch.from_numpy(
        val_label_global.astype(np.int64)
    ).to(device)

    train_onehot_global = torch.from_numpy(
        train_onehot_global.astype(np.int64)
    ).to(device)

    test_onehot_global = torch.from_numpy(
        test_onehot_global.astype(np.int64)
    ).to(device)

    val_onehot_global = torch.from_numpy(
        val_onehot_global.astype(np.int64)
    ).to(device)

    train_label_global_shift = train_label_global - 1

    # =========================================================
    # Input
    # =========================================================
    net_input = torch.from_numpy(
        np.asarray(data, dtype=np.float32)
    ).to(device)

    train_index = train_index.reshape(-1)
    val_index = val_index.reshape(-1)
    test_index = test_index.reshape(-1)

    train_index_global = train_index_global.reshape(-1)

    # =========================================================
    # Model
    # =========================================================
    model = MFS_HiNet.MFSCN(
        bands,
        class_num,
        class_num_global,
        layer, 
        kernel_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    zeros = torch.zeros(height * width).float().to(device)

    best_loss = float('inf')

    # =========================================================
    # Load Pretrained Weight
    # =========================================================
    if load_model:

        pretrained_dict = torch.load(
            path_weight + parent_model_name + ".pt"
        )

        model_dict = model.state_dict()

        remove_keys = []

        for key in pretrained_dict.keys():

            if 'hyperbolic_classifier' in key:
                remove_keys.append(key)

            if 'euc_classifier' in key:
                remove_keys.append(key)

            if 'fusion_classifier' in key:
                remove_keys.append(key)

        for key in remove_keys:
            del pretrained_dict[key]

        if use_parameter_inheritance:

            for key in pretrained_dict.keys():

                if key in model_dict:

                    old_param = pretrained_dict[key]
                    new_param = model_dict[key]

                    if old_param.shape == new_param.shape:

                        pretrained_dict[key] = (
                            inheritance_alpha * old_param +
                            (1 - inheritance_alpha) * new_param
                        )

        same_params = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict
        }

        model_dict.update(same_params)

        model.load_state_dict(model_dict)

        print('load model')

    # =========================================================
    # Train
    # =========================================================
    model.train()

    for epoch in range(max_epoch + 1):

        optimizer.zero_grad()

        (
            embedding_euc,
            embedding_hi,
            embedding_euc_global,
            curvature_tensor,
            output_euc,
            output_hi,
            output_global,
            output_fusion
        ) = model(net_input, curvature)

        # -----------------------------------------------------
        # Metric Loss
        # -----------------------------------------------------
        hi_loss = utils.bolicDistance_triloss(
            embedding_hi[train_index],
            train_label[train_index],
            curvature,
            hi_margin
        )

        euc_loss = utils.eucDistance_triloss(
            embedding_euc[train_index],
            train_label[train_index],
            euc_margin
        )

        euc_global_loss = utils.eucDistance_triloss(
            embedding_euc_global[train_index_global],
            train_label_global[train_index_global],
            euc_margin
        )

        # -----------------------------------------------------
        # Classification Loss
        # -----------------------------------------------------
        ce_hi = criterion(
            output_hi[train_index],
            train_label_shift[train_index].long()
        )

        ce_euc = criterion(
            output_euc[train_index],
            train_label_shift[train_index].long()
        )

        ce_global = criterion(
            output_global[train_index_global],
            train_label_global_shift[train_index_global].long()
        )

        ce_fusion = criterion(
            output_fusion[train_index],
            train_label_shift[train_index].long()
        )

        classification_loss = (
            ce_hi +
            ce_euc +
            ce_global +
            ce_fusion
        )

        metric_loss = (
            hi_loss +
            euc_loss +
            euc_global_loss
        )

        total_loss = classification_loss + metric_loss

        total_loss.backward()

        optimizer.step()

        # =====================================================
        # Validation
        # =====================================================
        if epoch % 10 == 0:

            with torch.no_grad():

                val_ce_hi = criterion(
                    output_hi[val_index],
                    val_label_shift[val_index].long()
                )

                val_ce_euc = criterion(
                    output_euc[val_index],
                    val_label_shift[val_index].long()
                )

                val_loss = val_ce_hi + val_ce_euc

                val_oa_hi = utils.evaluate_performance(
                    output_hi,
                    val_label,
                    val_onehot,
                    zeros
                )

                val_oa_euc = utils.evaluate_performance(
                    output_euc,
                    val_label,
                    val_onehot,
                    zeros
                )

                print(
                    f"Epoch: {epoch:04d} | "
                    f"Loss: {total_loss:.4f} | "
                    f"Val HI OA: {val_oa_hi:.4f} | "
                    f"Val EUC OA: {val_oa_euc:.4f}"
                )

                if val_loss < best_loss:

                    best_loss = val_loss

                    torch.save(
                        model.state_dict(),
                        path_weight + current_model_name + ".pt"
                    )

    torch.cuda.empty_cache()

    # =========================================================
    # Test
    # =========================================================
    with torch.no_grad():

        model.load_state_dict(
            torch.load(path_weight + current_model_name + ".pt")
        )

        model.eval()

        start = time.perf_counter()

        (
            embedding_euc,
            embedding_hi,
            embedding_euc_global,
            curvature_tensor,
            output_euc,
            output_hi,
            output_global,
            output_fusion
        ) = model(net_input, curvature)

        end = time.perf_counter()

        print(f"Inference Time: {end - start:.6f} s")

        output_pred = torch.argmax(output_fusion, dim=1)

        output_pred = np.array(output_pred.cpu())

        output_pred = label_recover(
            unique_numbers,
            output_pred
        )

        output_pred = torch.tensor(output_pred).to(device)

        output_pred = output_pred - 1
        print(output_pred.shape)
        print(test_label.shape)
        print(test_onehot.shape)
        test_oa = evaluate_performance(
            output_pred,
            test_label,
            test_onehot,
            zeros
        )

        print(f"Test OA: {test_oa:.4f}")

    torch.cuda.empty_cache()

    return test_oa, output_pred
def setup_seed(seed):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.benchmark = False 
def label_change(train_samples_gt):
    unique_numbers = np.unique(train_samples_gt)
    train_samples_gt_to1 = copy.deepcopy(train_samples_gt)
    number, = unique_numbers.shape
    for i in range (number):
        label = unique_numbers[i]
        index = np.array(np.where(train_samples_gt==label))[0]
        train_samples_gt_to1[index] = i
    return unique_numbers,train_samples_gt_to1
def label_recover(unique_numbers,train_samples_gt):
    train_samples_gt_to1 = copy.deepcopy(train_samples_gt)
    number, = unique_numbers.shape
    for i in range (1,number):
        label = unique_numbers[i]
        index = np.array(np.where(train_samples_gt==(i-1)))[0]
        train_samples_gt_to1[index] = label
    return train_samples_gt_to1




def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx = (train_samples_gt!=0).float()        # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()       # 有效标签的个数
        correct_prediction = torch.where(network_output ==torch.argmax(train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
        OA= correct_prediction.cpu() / available_label_count
        return OA



def Get_train_and_test_data(img_size, img,img_gt):
    H0, W0, C = img.shape
    if H0<img_size:
        gap = img_size-H0
        mirror_img = img[(H0-gap):H0,:,:]
        mirror_img_gt = img_gt[(H0-gap):H0,:]
        img = np.concatenate([img,mirror_img],axis=0)
        img_gt = np.concatenate([img_gt,mirror_img_gt],axis=0)
    if W0<img_size:
        gap = img_size-W0
        mirror_img = img[:,(W0 - gap):W0,:]
        mirror_img_gt = img_gt[(W0-gap):W0,:]
        img = np.concatenate([img,mirror_img],axis=1)
        img_gt = np.concatenate([img_gt,mirror_img_gt],axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H+1)*img_size - H
        mirror_img = img[(H - gap):H, :, :]
        mirror_img_gt = img_gt[(H - gap):H, :]
        img = np.concatenate([img, mirror_img], axis=0)
        img_gt = np.concatenate([img_gt,mirror_img_gt],axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap):W, :]
        mirror_img_gt = img_gt[:, (W - gap):W]
        img = np.concatenate([img, mirror_img], axis=1)
        img_gt = np.concatenate([img_gt,mirror_img_gt],axis=1)
        # gap = img_size - num_W*img_size
        # img = img[:,(W - gap):W,:]
    H, W, C = img.shape
    print('padding img:', img.shape)

    num_H = H // img_size
    num_W = W // img_size

    sub_imgs = []
    for i in range(num_H):
        for j in range(num_W):
            z = img[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :]
            sub_imgs.append(z)
    sub_imgs = np.array(sub_imgs)  # [num_H*num_W,img_size,img_size, C ]

    return sub_imgs, num_H, num_W,img_gt,img
def image_reshape(y,height,width,height_orgin,width_orgin,class_num):
    y = y.reshape(height,width,class_num)
    y= y[0:height_orgin,0:width_orgin,:]
    y= y.reshape(height_orgin*width_orgin,class_num)
    return y
def patch_reshape(pred,num_H, num_W, class_num, img_size):
    pred = torch.reshape(pred, [num_H, num_W, class_num, img_size, img_size])
    pred = torch.permute(pred, [2, 0, 3, 1, 4])  # [2,num_H, img_size,num_W, img_size]]
    pred = torch.reshape(pred, [class_num, num_H * img_size* num_W * img_size])
    pred = torch.permute(pred, [1, 0]) 
    return pred