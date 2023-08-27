import visdom
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import zipfile
import io
import os

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))
    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)
    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    r = np.linspace(0, len(feat), length+1, dtype=np.int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def modelsize(model, input, type_size=4):
    # check GPU utilisation
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(str(test_info["test_AUC"][-1]))
    fo.write('\n')
    fo.write(str(test_info["all_AUC"][-1]))
    fo.close()

def scorebinary(scores=None, threshold=0.5):
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold < threshold] = 0
    scores_threshold[scores_threshold >= threshold] = 1
    return scores_threshold

def my_anomap(video_predict, gt, video_name, save_path, itr, save_root, zip=False):
    """

    :param predict_dict:
    :param label_dict:
    :param save_path:
    :param itr:
    :param zip: boolen, whether save plots to a zip
    :return:
    """
    if os.path.exists(os.path.join(save_root, save_path, 'plot')) == 0:
        os.makedirs(os.path.join(save_root, save_path, 'plot'))
        
    gt_start_index = 0
    if zip:
        zip_file_name = os.path.join(save_root, save_path, 'plot', 'itr_{}.zip'.format(itr))
        with zipfile.ZipFile(zip_file_name, mode="w") as zf:
            
            for i in range(0,len(video_predict)):
                img_name = video_name[i] + '.jpg'
                predict_np = video_predict[i].repeat(16)
                num_frames = len(video_predict[i].repeat(16))
                label_np = gt[gt_start_index:gt_start_index+num_frames]
                gt_start_index = gt_start_index+num_frames
                x = np.arange(len(predict_np))
                plt.plot(x, predict_np, label='Anomaly scores', color='b', linewidth=1)
                plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
                plt.yticks(np.arange(0, 1.1, step=0.1))
                plt.xlabel('Frames')
                #plt.grid(True, linestyle='-.')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf)
                plt.close()
                zf.writestr(img_name, buf.getvalue())
            '''
            for k, v in predict_dict.items():
                img_name = k + '.jpg'
                predict_np = v.repeat(16)
                label_np = label_dict[k][:len(v.repeat(16))]
                x = np.arange(len(predict_np))
                plt.plot(x, predict_np, label='Anomaly scores', color='b', linewidth=1)
                plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
                plt.yticks(np.arange(0, 1.1, step=0.1))
                plt.xlabel('Frames')
                plt.grid(True, linestyle='-.')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf)
                plt.close()
                zf.writestr(img_name, buf.getvalue())
           '''


    else:
        for i in range(0,len(video_predict)):
            k = video_name[i]
            img_name = video_name[i] + '.jpg'
            predict_np = video_predict[i].repeat(16)
            num_frames = len(video_predict[i].repeat(16))
            label_np = gt[gt_start_index:gt_start_index+num_frames]
            gt_start_index = gt_start_index+num_frames
            x = np.arange(len(predict_np))
            plt.plot(x, predict_np, color='b', label='predicted scores', linewidth=1)
            plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xlabel('Frames')
            plt.ylabel('Anomaly Scores')
            #plt.grid(True, linestyle='-.')
            plt.ylim(0,None)
            plt.legend()
            plt.legend(loc='best')
            # plt.show()
            if os.path.exists(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))) == 0:
                os.makedirs(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr)))
                plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k+ '.pdf'), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k+ '.pdf'), bbox_inches='tight')
            plt.close()




def anomap(predict_dict, label_dict, save_path, itr, save_root, zip=False):
    """

    :param predict_dict:
    :param label_dict:
    :param save_path:
    :param itr:
    :param zip: boolen, whether save plots to a zip
    :return:
    """
    if os.path.exists(os.path.join(save_root, save_path, 'plot')) == 0:
        os.makedirs(os.path.join(save_root, save_path, 'plot'))
    if zip:
        zip_file_name = os.path.join(save_root, save_path, 'plot', 'itr_{}.zip'.format(itr))
        with zipfile.ZipFile(zip_file_name, mode="w") as zf:
            for k, v in predict_dict.items():
                img_name = k + '.jpg'
                predict_np = v.repeat(16)
                label_np = label_dict[k][:len(v.repeat(16))]
                x = np.arange(len(predict_np))
                plt.plot(x, predict_np, label='Anomaly scores', color='b', linewidth=1)
                plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
                plt.yticks(np.arange(0, 1.1, step=0.1))
                plt.xlabel('Frames')
                plt.grid(True, linestyle='-.')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf)
                plt.close()
                zf.writestr(img_name, buf.getvalue())


    else:
        for k, v in predict_dict.items():
            predict_np = v.repeat(16)
            label_np = label_dict[k][:len(v.repeat(16))]
            x = np.arange(len(predict_np))
            plt.plot(x, predict_np, color='b', label='predicted scores', linewidth=1)
            plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xlabel('Frames')
            plt.ylabel('Anomaly scores')
            plt.grid(True, linestyle='-.')
            plt.legend()
            # plt.show()
            if os.path.exists(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))) == 0:
                os.makedirs(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr)))
                plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
            else:
                plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
            plt.close()