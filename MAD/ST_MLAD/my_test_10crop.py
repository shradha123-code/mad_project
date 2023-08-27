import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, confusion_matrix
from utils import scorebinary, anomap, my_anomap
import numpy as np
import os
import pickle

print('In my test')

#sh_root_path = '/disk/zc/project/2021/bak/C3D_feature_extraction/C3D_Features_txt/Open_Filter/I3D/ST/new_data/t_0.75/'
sh_root_path = '/content/drive/MyDrive/VAD_Code/VAD_Code/t_0.75/'
ucf_root_path = '/content/drive/MyDrive/VAD_Code/VAD_Code/t_0.75/'
cs_root_path = '/content/drive/MyDrive/VAD_Code/VAD_Code/gt_workdir'


seen_or_open = 'open'

#label_dict_path = '/disk/zc/dataset/ST_I3D_Fea/dataset_divide/'
label_dict_path = '/content/drive/MyDrive/VAD_Code/VAD_Code/train_ckpt/'
cs

#with open(file=os.path.join(label_dict_path, 'video_label.pickle'), mode='rb') as f:
#    video_label_dict = pickle.load(f)

#with open(file=os.path.join(label_dict_path, 'st-i3d.pkl'), mode='rb') as f:
#     video_label_dict = pickle.load(f)

def eval_other_metric(itr, video_label, video_predict, gt, video_name, plot):
    #gt = np.load('/disk/zc/project/2021/bak/OpenSource/ICCV_21/RTFM-main/list/my-gt-ucf.npy')
    #video_predict = np.array(video_predict)
    gt.reshape(-1,1)
    print('gt shape:',gt.shape)
    
    all_predict_np = np.zeros(0)
    all_label_np = np.zeros(0)
    normal_predict_np = np.zeros(0)
    normal_label_np = np.zeros(0)
    abnormal_predict_np = np.zeros(0)
    abnormal_label_np = np.zeros(0)
    
    gt_start_index = 0
    for i in range(0,len(video_predict)):
        if video_label[i] == '1':
            num_frames = len(video_predict[i].repeat(16))
            frame_labels = gt[gt_start_index:gt_start_index+num_frames]
            #print('frame_labels:',frame_labels)
            gt_start_index = gt_start_index+num_frames
            all_predict_np = np.concatenate((all_predict_np, video_predict[i].repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels))
            abnormal_predict_np = np.concatenate((abnormal_predict_np, video_predict[i].repeat(16)))
            abnormal_label_np = np.concatenate((abnormal_label_np, frame_labels))
        elif video_label[i] == '0':
            
            num_frames = len(video_predict[i].repeat(16))
            frame_labels = gt[gt_start_index:gt_start_index+num_frames]
            #print('frame_labels:',frame_labels)
            gt_start_index = gt_start_index+num_frames
            all_predict_np = np.concatenate((all_predict_np, video_predict[i].repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels))
            normal_predict_np = np.concatenate((normal_predict_np, video_predict[i].repeat(16)))
            normal_label_np = np.concatenate((normal_label_np, frame_labels))
            
    all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)
    binary_all_predict_np = scorebinary(all_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=all_label_np, y_pred=binary_all_predict_np).ravel()
    all_ano_false_alarm = fp / (fp + tn)
    
    print('cal all_ano_miss_alarm')
    print('fn:',fn)
    print('tp+fn:', tp+fn)
    all_ano_miss_alarm = fn / (tp+fn)
    
    binary_normal_predict_np = scorebinary(normal_predict_np, threshold=0.5)
    # tn, fp, fn, tp = confusion_matrix(y_true=normal_label_np, y_pred=binary_normal_predict_np).ravel()
    fp_n = binary_normal_predict_np.sum()
    normal_count = normal_label_np.shape[0]
    normal_ano_false_alarm = fp_n / normal_count

    abnormal_auc_score = roc_auc_score(y_true=abnormal_label_np, y_score=abnormal_predict_np)
    binary_abnormal_predict_np = scorebinary(abnormal_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=abnormal_label_np, y_pred=binary_abnormal_predict_np).ravel()
    abnormal_ano_false_alarm = fp / (fp + tn)
    
    print('cal abnormal_ano_miss_alarm')
    print('fn:',fn)
    print('tp+fn:', tp+fn)
    abnormal_ano_miss_alarm = fn / (tp+fn)

    print('Iteration: {} AUC_score_all_video is {}'.format(itr, all_auc_score))
    print('Iteration: {} AUC_score_abnormal_video is {}'.format(itr, abnormal_auc_score))
    print('Iteration: {} ano_false_alarm_all_video is {}'.format(itr, all_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_normal_video is {}'.format(itr, normal_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_abnormal_video is {}'.format(itr, abnormal_ano_false_alarm))
    print('Iteration: {} all_ano_miss_alarm  is {}'.format(itr, all_ano_miss_alarm)) 
    print('Iteration: {} abnormal_ano_miss_alarm is {}'.format(itr, abnormal_ano_miss_alarm)) 
    
    save_root = 'st_res/'
    save_root = 'ucf_res/'
    score_save_path = 'open_score_img/'
    res_save_path = 'res/'
    if plot:
        my_anomap(video_predict, gt, video_name, score_save_path, itr, save_root, zip=False)
        
        with open(file=os.path.join(save_root, res_save_path, 'open_result.txt'), mode='a+') as f:
            f.write('itration_{}_AUC_Score_all_video is {}\n'.format(itr, all_auc_score))
            f.write('itration_{}_AUC_Score_abnormal_video is {}\n'.format(itr, abnormal_auc_score))
            f.write('itration_{}_ano_false_alarm_all_video is {}\n'.format(itr, all_ano_false_alarm))
            f.write('itration_{}_ano_false_alarm_normal_video is {}\n'.format(itr, normal_ano_false_alarm))
            f.write('itration_{}_ano_false_alarm_abnormal_video is {}\n'.format(itr, abnormal_ano_false_alarm))
            f.write('itration_{}_abnormal_ano_miss_alarm is {}\n'.format(itr, abnormal_ano_miss_alarm))

def test_all(dataloader, model, args, viz, device):
    
    #seen_gt = np.load(ucf_root_path + 'seen_ucf_gt.npy')
    #open_gt = np.load(ucf_root_path + 'open_ucf_gt.npy')
    
    #seen_pred = np.load('/disk/zc/project/2021/bak/OpenSource/ICCV_21/My_RTFM/ucf_res/' + 'seen_pred.npy')
    #seen_pred = np.load('/content/drive/MyDrive/VAD_Code/VAD_Code/ST_MLAD/st_res/' + 'seen_pred.npy')
    #seen_pred = np.load('/content/drive/MyDrive/VAD_Code/VAD_Code/ST_MLAD/ucf_res/' + 'seen_pred.npy')

    #open_pred = np.load('ucf_res/' + 'open_pred.npy')
    
    
    seen_gt = np.load(sh_root_path + 'seen_st_gt.npy')
    open_gt = np.load(sh_root_path + 'open_st_gt.npy')
    
    #seen_pred = np.load('/disk/zc/project/2021/bak/OpenSource/ICCV_21/My_RTFM/st_res/' + 'seen_pred.npy')
    seen_pred = np.load('/content/drive/MyDrive/VAD_Code/VAD_Code/ST_MLAD/st_res/' + 'seen_pred.npy')
    open_pred = np.load('st_res/' + 'open_pred.npy')
    
    
    #all_gt = np.concatenate((seen_gt,open_gt))
    #all_pred = np.concatenate((seen_pred,seen_pred))
    
    all_gt = np.concatenate((open_gt,seen_gt))
    all_pred = np.concatenate((open_pred,seen_pred))
    
    fpr, tpr, threshold = roc_curve(list(all_gt), all_pred)
    np.save('fpr.npy', fpr)
    np.save('tpr.npy', tpr)
    rec_auc = auc(fpr, tpr)
    print('all auc : ' + str(rec_auc))
    return rec_auc

def test(dataloader, model, args, viz, device):
    print('in test')
    print(args)
    if args.dataset == 'shanghai':
        #rgb_list_file ='/disk/zc/project/2021/bak/OpenSource/ICCV_21/RTFM-main/list/shanghai-i3d-test-10crop.list'
        rgb_list_file = sh_root_path + seen_or_open + '_st_test.list'
    elif args.dataset == 'ucf':
        #rgb_list_file ='/disk/zc/project/2021/bak/OpenSource/ICCV_21/RTFM-main/list/ucf-i3d-test.list'
        rgb_list_file = ucf_root_path + seen_or_open + '_ucf_test.list'
    
    file_list = list(open(rgb_list_file))
    o_video_label = [0] * len(file_list)
    o_video_predict = []
    o_video_name = []
    
    for v_count in range(0,len(file_list)):
        file = file_list[v_count]
        p,f = os.path.split(file)
        print(f)
        f = f[0:-9]
        v_name = f
        #print('v_name:',v_name)
        o_video_name.append(v_name)
        #if video_label_dict[f] == [0.]:
        #    o_video_label[v_count] = '0'
        #elif video_label_dict[f] == [1.]:
        #    o_video_label[v_count] = '1'
            
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            #print('******')
            #print('input shape:',input.shape)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            v_pred = list(sig.cpu().detach().numpy())
            v_pred = np.array(v_pred)
            o_video_predict.append(v_pred)
            #print(sig)
            pred = torch.cat((pred, sig))
            #print(pred)
            print('sig shape:',sig.shape)
        
        print('final i ',i)

        if args.dataset == 'shanghai':
            gt = np.load(sh_root_path + 'open_st_gt.npy')
            #print(gt)
            print(np.shape(gt))
            save_path = 'st_res/' + 'open_pred.npy'
        else:
            gt = np.load(ucf_root_path + 'open_ucf_gt.npy')
            save_path = 'ucf_res/' + 'open_pred.npy'

        #print('gt shape:',gt.shape)
        #gt.reshape(-1,1)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        print('pred shape:',pred.shape)
        #pred.reshape(-1,1)
        #fpr, tpr, threshold = roc_curve(list(gt), pred[0:59552])
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        #print('auc : ' + str(rec_auc))
        np.save(save_path, pred)
        
        return rec_auc

