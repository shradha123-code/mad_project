import argparse


parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
parser.add_argument('--rgb-list', default='', help='list of rgb features ')
parser.add_argument('--test-rgb-list', default= '', help='list of test rgb features ')
parser.add_argument('--gt', default= '', help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
parser.add_argument('--lr', type=str, default='[0.001]*20000', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=4, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--model-name', default='rtfm', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--dataset', default='shanghai', help='dataset to train on (default: )')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=15000, help='maximum iteration to train (default: 100)')
parser.add_argument('--shRootDir', type=str, default='/mnt/c/code/mad_project/MAD/t_0.75', help='shanghai root')
parser.add_argument('--ucfRootDir', type=str, default='/mnt/c/code/mad_project/MAD/t_0.75', help='ucf root')
parser.add_argument('--csRootDir', type=str, default='/mnt/c/code/mad_project/MAD/t_0.75', help='cs root')
parser.add_argument('--resPath', type=str, default='/mnt/c/code/mad_project/MAD/ST_MLAD/st_res2', help='result dir')
parser.add_argument('--chkptPath', type=str, default='/mnt/c/code/mad_project/MAD/train_ckpt', help='checkpoint dir')
parser.add_argument('--tempDir', type=str, default='/mnt/c/code/mad_project/MAD/ST_MLAD/temp', help='temp dir')
parser.add_argument('--csfilelist', type=str, default='open_cs_test.list', help='paths of features npy of cs dataset')
parser.add_argument('--csgt', type=str, default='gt_cs.npy', help='cs ground truth')
parser.add_argument('--savepred', type=str, default='open_cspred.npy', help='temp dir')
parser.add_argument('--trainTempDir', type=str, default='/mnt/c/code/mad_project/MAD/ST_MLAD/all_train/tmp_dir', help='temp dir')
parser.add_argument('--trainLabelDir', type=str, default='/mnt/c/code/mad_project/MAD/ST_MLAD/all_train/label_dir', help='temp dir')