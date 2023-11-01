import os
import pandas as pd
import numpy as np
import shutil

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

### Ground truth generation with excel file #####
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


# Function to generate text files for meta training. The output is a list of filepaths
# for either normal or abnormal classes with its associated label. In training, normal videos are divided into
# K classes and abnormal into L classes. So, if there are N_n normal videos and N_a abnormal videos, then 
# there will be N_n/K and N_a/L normal and abnormal videos per class respectively, which will share the
# same class label. For example, if there are 20 normal videos and we have K = 5 normal classes, then 4 videos 
# per class will share the same label. The labels will be 0,1,2,3,4. Hence in the nor class label file,
# there will be addresses of all these 20 videos, but next to each video will be its class label.
# 4 videos will share the same label, then then next 4 share the same and so on.

# Input: inPath -> global path of folder where npy feature files exist
# numClass: Number of classes
# firstLabel: Label of first class.
# outFolder: Output file folder
# outFileName: output file name

def generate_class_labels(inPath, numClass, firstLabel, outFolder, outFileName):
  outFilePath = os.path.join(outFolder, outFileName)
  inFileList = os.listdir(inPath)
  numInFiles = len(inFileList)
  with open(outFilePath, 'w') as fp:
    videosPerClass = np.ceil(numInFiles/numClass)
    label = firstLabel
    for i in range(0,numInFiles):
      addr1 = os.path.join(inPath,inFileList[i]) + ' ' + str(label)
      print(addr1)
      fp.write("%s\n" % addr1)
      if (i+1)%videosPerClass==0:
        label = label + 1


def copy_files(inDir,labelFile,outDir):
  fList = os.listdir(inDir) # 
  with open(labelFile,'r') as f:
    lines = [line.rstrip() for line in f]
  f.close()
  #numLines = len(lines)
  print(lines)
  outList = []
  for line in lines:
    line1 = line.split()
    line1 = line1[0]
    inFilePath = os.path.join(inDir,line1)
    #inFilePath = str(inFilePath) + '\n'
    print(inFilePath)
    shutil.move(inFilePath,outDir)
    #outList.append(inFilePath)
  #with open(outFile,'w+') as fOut:
  #  fOut.writelines(outList)
  
  ### Ground truth generation for ucf with text file  #####
def gen_ucf_ground_truth(inputAnnotFile1, outDir, outFile):
   with open(inputAnnotFile1,'r') as f2:
    lines2 = [line.rstrip() for line in f2]
    f2.close()
  #df = pd.read_excel(inputAnnotFile)
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


#### Run 100 times



