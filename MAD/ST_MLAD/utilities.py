import os
import pandas as pd
import numpy as np

def gen_npy_address(path, outDir, outFile):
   #path = "/content/drive/MyDrive/VAD_Code/VAD_Code/chain_cropped_newFea"
   dir_list = os.listdir(path)
   li=[]
   outPath = os.path.join(outDir,outFile)
   print(outPath)
   for files in dir_list:
    addr=(os.path.join(path, files))
    li.append(addr)
    print(li)
   with open(outPath,'w') as f1:
     f1.write('\n'.join(li))

def gen_ground_truth(inputAnnotFile, outDir, outFile):
  df = pd.read_excel(inputAnnotFile)
  numFiles = df.shape[0]
  gt = []
  countFrame = 0
  totalFrames = 0
  outPath = os.path.join(outDir,outFile)
  for i in range(numFiles):
    fName = df['video name'][i]
    features = np.load(fName.strip('\n'), allow_pickle=True)
    features = np.array(features, dtype=np.float32)
    print(features.shape)
    
    numFrames = features.shape[0] * 16
    print('i, numFrames: ',i,numFrames)
    totalFrames += numFrames
    
    isAbnormal = df['isAbnormal'][i]
    if (isAbnormal):
      print('startframe', df['start frame'][i], i)
      startFrame = int(df['start frame'][i])
      endFrame = int(df['end frame'][i])
      if (endFrame > numFrames):
        print('Endframe is greater, i, endFrame, numFrames',i,endFrame,numFrames)
      for i in range(0,startFrame):
        gt.append(0)
        countFrame += 1
      if not ((endFrame + 1) > numFrames):
        for i in range(startFrame,endFrame + 1):
          gt.append(1)
          countFrame += 1
        for i in range(endFrame + 1, numFrames):
          gt.append(0)
          countFrame += 1
      else:
        for i in range(startFrame,numFrames):
          gt.append(1)
          countFrame += 1
    else:
      for i in range(0,numFrames):
        gt.append(0)
        countFrame += 1
    print('countFrame is ',countFrame)

  #output_file = 'gt_cs.npy'
  gt = np.array(gt, dtype=float)
  np.save(outPath, gt)
  print(len(gt))

  print(gt)
  print('TotalFrames: ',totalFrames)
