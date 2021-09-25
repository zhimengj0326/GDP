import math
import numpy as np
import torch
from kernel_estimator import nadarayaWatsonEstomator

def eval_fairness_hard(src, dst, ts, label, SHn_feat, BATCH_SIZE, lr_model, tgan, num_layer):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []   

    pred_prob = np.zeros(len(src))
    num_test_instance = len(src)
    num_test_batch = math.ceil(num_test_instance / BATCH_SIZE)

    shfeature = SHn_feat["shfeature"] # num_nodes * 2
    s_node_index = SHn_feat["index"]  # (num_nodes,)  num_nodes=4287

    sum_SP_T, count_SP_T = 0, 0
    sum_SP_F, count_SP_F = 0, 0
    sum_EO_T, count_EO_T = 0, 0
    sum_EO_F, count_EO_F = 0, 0
    with torch.no_grad():
        lr_model.eval()
        tgan = tgan.eval()
        # print(f'src={src.shape}')
        # print(f'dst={dst}')
        # print(f'ts={ts}')
        # print(f'label={label}')
        # print(f'num_test_instance={num_test_instance}')
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]
            src_label = label[src_l_cut]

            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)            
            # src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()


        T_flag = (shfeature[src, 0] > 0.) \
                * s_node_index[src] 
        F_flag = (shfeature[src, 0] == 0) \
                * s_node_index[src]

        # print(f'T_flag={T_flag}')
        # print(f'T_flag={shfeature[src_l, 0] > 0.}')
        pred_score = pred_prob
        pred_label = pred_score > 0.5
        true_label = label
        ## select samples with y=1
        label_index = (true_label[src] > 0) \
                    * s_node_index[src]

        SP_T = pred_score * T_flag
        SP_F = pred_score * F_flag              
        count_SP_T = np.sum(T_flag)
        count_SP_F = np.sum(F_flag)

        # print(f'shfeature[src_l, 0]={shfeature[src_l, 0]}')
        # print(f'pred_score={pred_score}')
        # print(f'true_label={true_label}')
        # print(f'SP_T={SP_T}')
        # print(f'count_SP_F={count_SP_F}')
        # print(f'label_index={label_index}')

        EO_T = SP_T[label_index]
        EO_F = SP_F[label_index]
        count_EO_T = np.sum(T_flag[label_index])
        count_EO_F = np.sum(F_flag[label_index])        

        D_SP = abs(np.sum(SP_T)/float(count_SP_T) - np.sum(SP_F)/float(count_SP_F))
        D_EO = abs(np.sum(EO_T)/float(count_EO_T) - np.sum(EO_F)/float(count_EO_F))
    # print(f'D_SP1={np.sum(SP_F)/float(count_SP_F)}') 
    # print(f'D_SP={D_SP}')
    # print(f'D_EO={D_EO}')
    return np.around(D_SP, 4), np.around(D_EO, 4)


def eval_fairness_kernel(src, dst, ts, label, Sn_feat, BATCH_SIZE, lr_model, tgan, num_layer):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []   

    sfeature = Sn_feat["sfeature"] # num_nodes * 2
    s_node_index = Sn_feat["index"]  # (num_nodes,)  num_nodes=4287

    feature_solution = 20
    kernelFunction = 'cosinus'
    h = 1

    
    pred_prob = np.zeros(len(src))
    num_test_instance = len(src)
    num_test_batch = math.ceil(num_test_instance / BATCH_SIZE)

    with torch.no_grad():
        lr_model.eval()
        tgan = tgan.eval()

        pred_estimator = np.empty((feature_solution,0))
        pred_estimator2 = np.empty((feature_solution,0))

        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]
            label_l_cut = label[src_l_cut]
            size = len(src_l_cut)

            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)            
            # src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()

            Flag = s_node_index[src_l_cut]
            dist = sfeature[src_l_cut][Flag,0] / 100
            # print(f'dist={dist}')
            

            # dist = 1/2 * np.linalg.norm(sfeature[src_l_concat][Flag] - sfeature[dst_l_concat][Flag], ord=1, axis=1)
            # dist = dist / np.max(dist)

            # T_flag = T_flag * s_node_index[s_idx:e_idx]
            # F_flag = F_flag * s_node_index[s_idx:e_idx]


            pred_score = pred_prob[s_idx:e_idx][Flag]
            # print(f'pred_score={pred_score.shape}')
            

            ## SP
            beta = np.expand_dims(np.arange(feature_solution)/feature_solution, axis=1)

            # print(f'Flag={Flag}')
            # print(f'pred_score={pred_score.shape}')

            # print(f'beta - dist={beta - dist}')
            # print(f'dist={dist}')
            # print(f'beta={beta.shape}')
            
            pred_estimator_batch, weight = nadarayaWatsonEstomator(beta - dist, pred_score, kernelFunction, h, scaleKernel=True, \
                                derivative=0)  ### feature_solution * test_batch
            # print(f'pred_estimator_batch={pred_estimator_batch}')
            
            pred_estimator = np.concatenate((pred_estimator, np.expand_dims(pred_estimator_batch, axis=1)), axis=1)

            ## EO
            positive_index = np.argwhere(label_l_cut[Flag] > 0)
            # print(f'positive_index={positive_index.shape}')
            pred_score_2 = pred_score[positive_index].squeeze(-1)
            dist_2 = dist[positive_index].squeeze(-1)
            # print(f'dist_2={dist_2.shape}')
            # print(f'pred_score={pred_score.shape}')
            # print(f'beta - dist_2={beta - dist_2}')

            
            pred_estimator2_batch, weight_2 = nadarayaWatsonEstomator(beta - dist_2, pred_score_2, kernelFunction, h, scaleKernel=True, \
                                derivative=0)
            # print(f'pred_estimator_batch_2={pred_estimator2_batch.shape}')
            pred_estimator2 = np.concatenate((pred_estimator2, np.expand_dims(pred_estimator2_batch, axis=1)), axis=1)

    pred_estimator = np.mean(pred_estimator, axis=1)
    pred_estimator2 = np.mean(pred_estimator2, axis=1)

    pred_mean = np.sum(pred_estimator * weight) / feature_solution
    D_SP = np.sum(np.abs(pred_estimator - pred_mean) * weight) / feature_solution

    # print(f'weight={weight}')
    # print(f'pred_mean={pred_mean}')
    # print(f'pred_estimator={pred_estimator}')

    pred_mean_2 = np.sum(pred_estimator2 * weight_2) / feature_solution
    D_EO = np.sum(np.abs(pred_estimator2 - pred_mean_2) * weight_2) / feature_solution
                   

    # D_EO = abs(sum_EO_T/float(count_EO_T) - sum_EO_F/float(count_EO_F))
    # print(f'D_SP={D_SP}')
    # print(f'D_EO={D_EO}')
    return np.around(D_SP, 4), np.around(D_EO, 4)