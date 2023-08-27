from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
#from train import train
from train import *
from my_test_10crop import test,test_all
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
import pickle


print('here')

viz = Visualizer(env='ucf tech 10 crop', use_incoming_socket=False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

torch.backends.cudnn.enabled=True

#checkpoint_PATH = 'train_ckpt/st-i3d.pkl'
checkpoint_PATH = '/content/drive/MyDrive/VAD_Code/VAD_Code/train_ckpt/st-i3d.pkl'
                              
def load_checkpoint(model, checkpoint_PATH):
    model.load_state_dict(torch.load(checkpoint_PATH))
    return model
    
if __name__ == '__main__':
    args = option.parser.parse_args()
    print('hello')
    config = Config(args)

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    print('test_loader len is ',len(test_loader))

    model = Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    #optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)
    
    model = load_checkpoint(model, checkpoint_PATH)
    #print(model)
    #for param_tensor in model.state_dict():
    #  print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print('Main args are ',args)
    auc = test(test_loader, model, args, viz, device)
    all_auc = test_all(test_loader, model, args, viz, device)
    print('all auc : ' + str(all_auc))
    

