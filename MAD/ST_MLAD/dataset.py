import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')


#sh_root_path = '/disk/zc/project/2021/bak/C3D_feature_extraction/C3D_Features_txt/Open_Filter/I3D/ST/new_data/t_0.75/'
sh_root_path = '/content/drive/MyDrive/VAD_Code/VAD_Code/t_0.75/'
#sh_tmp_dir = '/disk/zc/project/2021/bak/C3D_feature_extraction/C3D_Features_txt/Open_Filter/I3D_ST/open_maml/tmp_train_eval/'
sh_tmp_dir = '/content/drive/MyDrive/VAD_Code/VAD_Code/ST_MLAD/temp/'
ucf_tmp_dir = '/content/drive/MyDrive/VAD_Code/all_train/tmp_dir/'

class MLAD_Dataset(data.Dataset):
    def __init__(self, args, is_meta_train, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.is_meta_train = is_meta_train
        self.dataset = args.dataset
        if self.dataset == 'shanghai':
            if test_mode:
                self.sh_test_rgb_list_file = sh_root_path + 'open_st_test.list'
            else:
                self.sh_meta_train_nor_rgb_list_file = sh_tmp_dir + 'meta_train/train_nor_list.list'
                self.sh_meta_train_abnor_rgb_list_file = sh_tmp_dir + 'meta_train/train_abnor_list.list'
                self.sh_meta_eval_nor_rgb_list_file = sh_tmp_dir + 'meta_eval/eval_nor_list.list'
                self.sh_meta_eval_abnor_rgb_list_file = sh_tmp_dir + 'meta_eval/eval_abnor_list.list'
                
                
                
        else:
            if test_mode:
                self.ucf_test_rgb_list_file = ucf_root_path + 'open_ucf_test.list'
            else:
                self.ucf_meta_train_nor_rgb_list_file = ucf_tmp_dir + 'meta_train/train_nor_list.list'
                self.ucf_meta_train_abnor_rgb_list_file = ucf_tmp_dir + 'meta_train/train_abnor_list.list'
                self.ucf_meta_eval_nor_rgb_list_file = ucf_tmp_dir + 'meta_eval/eval_nor_list.list'
                self.ucf_meta_eval_abnor_rgb_list_file = ucf_tmp_dir + 'meta_eval/eval_abnor_list.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal: 
                    if self.is_meta_train: # meta train
                        self.list = list(open(self.sh_meta_train_nor_rgb_list_file))
                        print('normal list for shanghai tech meta train')
                        #print(self.list)
                    else: # meta eval
                        self.list = list(open(self.sh_meta_eval_nor_rgb_list_file))
                        #print('normal list for shanghai tech meta eval')
                        #print(self.list)
                        
                else: 
                    if self.is_meta_train: #meta train
                        self.list = list(open(self.sh_meta_train_abnor_rgb_list_file))
                        print('abnormal list for shanghai tech meta train')
                        #print(self.list)
                    else: #meta eval
                        self.list = list(open(self.sh_meta_eval_abnor_rgb_list_file))
                        print('abnormal list for shanghai tech meta eval')
                        #print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal: 
                    if self.is_meta_train: # meta train
                        self.list = list(open(self.ucf_meta_train_nor_rgb_list_file))
                        print('normal list for ucf meta train')
                        #print(self.list)
                    else: # meta eval
                        self.list = list(open(self.ucf_meta_eval_nor_rgb_list_file))
                        print('normal list for ucf meta eval')
                        #print(self.list)
                        
                else: 
                    if self.is_meta_train: #meta train
                        self.list = list(open(self.ucf_meta_train_abnor_rgb_list_file))
                        print('abnormal list for ucf meta train')
                        #print(self.list)
                    else: #meta eval
                        self.list = list(open(self.ucf_meta_eval_abnor_rgb_list_file))
                        print('abnormal list for ucf meta eval')
                        #print(self.list)
                        

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame



class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = sh_root_path + 'open_st_test.list'
            else:
                self.rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
        else:
            if test_mode:
                self.rgb_list_file = ucf_root_path + 'open_ucf_test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        print(self.list)
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
