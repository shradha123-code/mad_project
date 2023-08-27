from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model, MLAD_Model
from dataset import Dataset, MLAD_Dataset
#from train import train
from train import *
from meta_train import meta_train
from my_test_10crop import test,test_all
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
import random
import os
import shutil
import pickle


viz = Visualizer(env='st tech 10 crop', use_incoming_socket=False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

torch.backends.cudnn.enabled=True

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            shutil.rmtree(c_path,True)
        else:
            os.remove(c_path)
            
def file_list_all(file_path):
    fp = open(file_path, "r")
    file_list = []
    for lines in fp.readlines():
        lines = lines.replace("\n","")
        file_list.append(lines)
    fp.close()
    return file_list  
            
def get_meta_train_list(class_index, tmp_dir, label_dir):
    nor_label_file = label_dir + 'nor_path_label.txt'
    abnor_label_file = label_dir + 'abnor_path_label.txt'
    
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    meta_train_dir = tmp_dir + 'meta_train/'
    
    # This meta_train_dir is created for each ordered pair in meta train.
    # First component of class_index is nor label, second is abnor_label
    # If meta_train_dir was created previously, it is deleted to create
    # txt files of feature paths of the current ordered pair.
    if not os.path.exists(meta_train_dir):
        os.mkdir(meta_train_dir)
    else:
        del_file(meta_train_dir)
        
    nor_train_index = class_index[0]
    abnor_train_index = class_index[1]
    nor_label_list = file_list_all(nor_label_file)
    abnor_label_list = file_list_all(abnor_label_file)
    #print(type(nor_label_list[0]))
    #print(nor_label_list[0].split())
    #print(nor_label_list[0].split()[1])
    meta_train_nor = []
    meta_train_abnor = []
    
    for i in range(0,len(nor_label_list)):
        # Make this more efficient. Right now, all lines are being 
        # checked, wasting time. Instead, introduce if condition to check
        # split_nor_label == nor_train_index, right after first line.
        # Only if this condition is satisfied, read the path. PUT IN 
        # PAPER !!!!! Say, this increased efficiency of the github code.
        split_nor_label = nor_label_list[i].split()[1]
        split_nor_path = nor_label_list[i].split()[0]
        print('split_nor_path:',split_nor_path)

        # COMMENTING OUT BELOW LINES SINCE WE HAVE PATH
        #fea_file_name = split_nor_path.split("/")[7]
        #print('fea_file_name:',fea_file_name)
        #fea_file_name = fea_file_name[7:]
        #fea_path = fea_dir + fea_file_name
        #print('fea_file_name:',fea_file_name)

        fea_path = split_nor_path
        
        if split_nor_label == nor_train_index:
            meta_train_nor.append(fea_path)

    with open(meta_train_dir + 'train_nor_list.list', 'w+') as train_nor_list:
        for fea in meta_train_nor:
            newline = fea+'\n'
            train_nor_list.write(newline)
    
    
    for i in range(0,len(abnor_label_list)):
        split_abnor_label = abnor_label_list[i].split()[1]
        split_abnor_path = abnor_label_list[i].split()[0]
        #fea_file_name = split_abnor_path.split("/")[7]
        #fea_path = fea_dir + fea_file_name
        #print('fea_file_name:',fea_file_name)

        fea_path = split_abnor_path

        if split_abnor_label == abnor_train_index:
            meta_train_abnor.append(fea_path)   
            
    with open(meta_train_dir + 'train_abnor_list.list', 'w+') as train_abnor_list:
        for fea in meta_train_abnor:
            newline = fea+'\n'
            train_abnor_list.write(newline)

def get_meta_eval_list(eval_index, tmp_dir, label_dir):
    nor_label_file = label_dir + 'nor_path_label.txt'
    abnor_label_file = label_dir + 'abnor_path_label.txt'
    
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    meta_eval_dir = tmp_dir + 'meta_eval/'
        
    if not os.path.exists(meta_eval_dir):
        os.mkdir(meta_eval_dir)
    else:
        del_file(meta_eval_dir)
    
    nor_eval_index = [eval_index[0], eval_index[1]]
    abnor_eval_index = [eval_index[2], eval_index[3]]
    nor_label_list = file_list_all(nor_label_file)
    abnor_label_list = file_list_all(abnor_label_file)
    #print(type(nor_label_list[0]))
    #print(nor_label_list[0].split())
    #print(nor_label_list[0].split()[1])

    meta_eval_nor = []
    meta_eval_abnor = []
    
    for i in range(0,len(nor_label_list)):
        split_nor_label = nor_label_list[i].split()[1]
        split_nor_path = nor_label_list[i].split()[0]
        #print('split_nor_path:',split_nor_path)
        #fea_file_name = split_nor_path.split("/")[7]
        #print('fea_file_name:',fea_file_name)
        #fea_file_name = fea_file_name[7:]
        #fea_path = fea_dir + fea_file_name
        fea_path = split_nor_path
        if split_nor_label == nor_eval_index[0] or split_nor_label == nor_eval_index[1] :
            meta_eval_nor.append(fea_path)
            
    #with open(meta_eval_dir + 'eval_nor_list.list', 'wb') as eval_nor_list:
    with open(meta_eval_dir + 'eval_nor_list.list', 'w+') as eval_nor_list:
        for fea in meta_eval_nor:
            newline = fea+'\n'
            eval_nor_list.write(newline)
        
        
    
    for i in range(0,len(abnor_label_list)):
        split_abnor_label = abnor_label_list[i].split()[1]
        split_abnor_path = abnor_label_list[i].split()[0]
        #fea_file_name = split_abnor_path.split("/")[7]
        #fea_path = fea_dir + fea_file_name
        #print('fea_path:',fea_path)
        fea_path = split_abnor_path
        if split_abnor_label == abnor_eval_index[0] or split_abnor_label == abnor_eval_index[1]:
            meta_eval_abnor.append(fea_path)

    with open(meta_eval_dir + 'eval_abnor_list.list', 'w+') as eval_abnor_list:
        for fea in meta_eval_abnor:
            newline = fea+'\n'
            eval_abnor_list.write(newline)
    

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    #iterations = 20000
    iterations = 5
    meta_step_size = 0.001
    stop_gradient = False
    meta_val_beta = 0.1
    #train_res_dir = 'ST/I3D/' + str(iterations) + '/'
    train_res_dir = 'train_check'
    output_path = '/content/drive/MyDrive/VAD_Code/all_train/output_train'  # put your own path here
    model_save_pth = '/content/drive/MyDrive/VAD_Code/all_train/model_train'
    test_info = {"epoch": [], "test_AUC": [], "all_AUC": []}
    #divide meta train and meta eval
    #tmp_dir = '/disk/zc/project/2021/bak/C3D_feature_extraction/C3D_Features_txt/Open_Filter/I3D_ST/open_maml/tmp_train_eval/'
    #label_dir = '/disk/zc/project/2021/bak/C3D_feature_extraction/C3D_Features_txt/Open_Filter/I3D/ST/pre_process_res/'
    #fea_dir = '/disk/zc/dataset/ST_I3D_Fea/Train/'

    tmp_dir = '/content/drive/MyDrive/VAD_Code/all_train/tmp_dir/'
    label_dir = '/content/drive/MyDrive/VAD_Code/all_train/label_dir/'
    #fea_dir 
    
    train_nor_class = ['0','1','2','3','4']
    train_abnor_class = ['5','6','7','8'] 
    
    #test_loader = DataLoader(Dataset(args, test_mode=True),
    #                          batch_size=1, shuffle=False,
    #                          num_workers=0, pin_memory=False)
    
    nor_abnor_class = []
    for i in range (len(train_nor_class)):
        list_em = [] 
        list_em.append(train_nor_class[i])
        list_tmp = list_em.copy()
        #print('list_tmp:',list_tmp)
        for j in range (len(train_abnor_class)):
            
            list_em.append(train_abnor_class[j])
        
            '''nor_abnor_class contains ordered pairs of one normal
            and one abnormal index'''
            nor_abnor_class.append(list_em)
            #print('list_em:',list_em)
            list_em = list_tmp.copy()
            #print('list_em:',list_em)
    print('nor_abnor_class:',nor_abnor_class)
    print('nor_abnor_class length:',len(nor_abnor_class))
    
    model = MLAD_Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)
    
    best_AUC = -1
    best_all_AUC = -1
    #output_path = 'my_output/' + train_res_dir   # put your own path here
    #model_save_pth = 'train_models/' + train_res_dir
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(model_save_pth):
        os.mkdir(model_save_pth)
        
    
    
       
    for step in range(1,iterations+1):
        meta_train_loss = 0.0
        # List of ordered pairs of nor-abnor class for meta train
        meta_train_class_index_list = []
        
        eval_nor_class = random.sample(train_nor_class,k=2)
        #print('meta eval_nor_class:',eval_nor_class)
        eval_abnor_class = random.sample(train_abnor_class,k=2)
        #print('meta eval_abnor_class:',eval_abnor_class)
        # eval_class_index contains normal and abnormal labels for meta test
        eval_class_index = eval_nor_class + eval_abnor_class
        print('meta eval_class_index:',eval_class_index)
        
        # Loop over ordered pairs of nor-abnor labels
        for j in range(0,len(nor_abnor_class)):
            # If any of the nor or abnor class for meta test is not found,
            # use that ordered pair for meta train
            if eval_class_index[0] not in nor_abnor_class[j] and eval_class_index[1] not in nor_abnor_class[j] \
            and eval_class_index[2] not in nor_abnor_class[j] and eval_class_index[3] not in nor_abnor_class[j]:
                meta_train_class_index_list.append(nor_abnor_class[j])
        
        print('meta_train_class_index_list:',meta_train_class_index_list)
        for index in range(0,len(meta_train_class_index_list)):
            get_meta_train_list(meta_train_class_index_list[index], tmp_dir, label_dir)
            
            meta_train_nloader = DataLoader(MLAD_Dataset(args, test_mode=False, is_meta_train=True, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True,generator=torch.Generator(device='cuda'))
                               
            meta_train_aloader = DataLoader(MLAD_Dataset(args, test_mode=False, is_meta_train=True, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True,generator=torch.Generator(device='cuda'))
                           
            if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.lr[step - 1]
            
            print('The len of meta_train_nloader:',len(meta_train_nloader))    
            print('The len of meta_train_aloader:',len(meta_train_aloader))  
            
            #if (step - 1) % len(meta_train_nloader) == 0:
            loadern_iter = iter(meta_train_nloader)
            print('The len of loadern_iter:',len(loadern_iter))
            print('loadern iter: ',loadern_iter)

            #if (step - 1) % len(meta_train_aloader) == 0:
            loadera_iter = iter(meta_train_aloader)
            print('The len of loadera_iter:',len(loadera_iter))
            print('loadern iter: ',loadera_iter)
                               
            loss = meta_train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device, \
                                meta_loss=None, meta_step_size=None, stop_gradient=False)
            meta_train_loss += loss
       
        get_meta_eval_list(eval_class_index, tmp_dir, label_dir)
        
        meta_eval_nloader = DataLoader(MLAD_Dataset(args, test_mode=False, is_meta_train=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True,generator=torch.Generator(device='cuda'))
            
        meta_eval_aloader = DataLoader(MLAD_Dataset(args, test_mode=False, is_meta_train=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True,generator=torch.Generator(device='cuda'))
                               
        loadern_iter = iter(meta_eval_nloader)
        
        loadera_iter = iter(meta_eval_aloader)
        
       
        meta_val_loss = meta_train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device,\
                                meta_loss=meta_train_loss, meta_step_size= meta_step_size, stop_gradient= stop_gradient)
        
        total_loss = meta_train_loss + meta_val_loss * meta_val_beta

        # init the grad to zeros first
        optimizer.zero_grad()

        # backward your network
        total_loss.backward()

        # optimize the parameters
        optimizer.step()
        print(
                'step:', step,
                'meta_train_loss:', meta_train_loss.item(),
                'meta_val_loss:', meta_val_loss.item())
                
    torch.save(model.state_dict(), model_save_pth  + args.model_name + 'final.pkl')
    
            
            
    
   
    

