'''
Author: BHM-Bob G 2262029386@qq.com
Date: 2022-02-22 20:41:43
LastEditors: BHM-Bob
LastEditTime: 2022-10-19 23:07:25
Description: 
'''
import csv
import glob
import os
import shutil


def MpPrint(mp,*Args):
    if mp == None:
        for item in Args:
            print(item,' ',end = '')
    else:
        mp.mprint(Args)
    print('')

def DelRepeatingFile(toDelRoot, referenceRoot, mp = None):
    """
    将toDelRoot中出现的与referenceRoot重复文件删除
    """

    toDelPaths = glob.glob(toDelRoot)
    toRemainPaths = glob.glob(referenceRoot)

    toDelNames = [ path.split(os.path.sep)[-1] for path in toDelPaths ]
    toRemainNames = [ path.split(os.path.sep)[-1] for path in toRemainPaths ]

    sumToDelPaths, sumToRemainPaths = len(toDelPaths), len(toRemainPaths)
    MpPrint(mp,sumToDelPaths,sumToRemainPaths)

    for idx, toDelName in enumerate(toDelNames):
        if toDelName in toRemainNames:
            MpPrint(mp,sumToDelPaths,sumToRemainPaths)
            try:
                os.remove(toDelPaths[idx])
                MpPrint(mp,"success to del",toDelName)
            except:
                MpPrint(mp,"unable to del",toDelName)
            if idx % 10 == 0:
                MpPrint(mp,idx,'/',sumToDelPaths)

#DelRepeatingFile(r"D:\AI\DataSet\Seq2ImgFluently\seq\Seq\Pos PreEmbIdx Seq\*.npy",r"D:\AI\DataSet\Seq2ImgFluently\seq\Seq\WordIdx Seq\*.npy")

def ReShapeDataSet():
    sourceDir = r"D:\AI\DataSet\102Flowers\source"
    torchDir = r"D:\AI\DataSet\102Flowers\torch"
    imgPaths = glob.glob(os.path.join(sourceDir,'*.jpg'))
    # imgPaths[idx] == os.path.join(sourceDir,'image_{0:5d}.jpg'.format(imgNums[idx]))
    imgNums = [ int( imgPath.split(os.path.sep)[-1].split('.')[0].replace('image_','') ) for imgPath in imgPaths ]
    imgLabels =[]
    with open(r"D:\AI\DataSet\102Flowers\labels.csv",mode = 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            imgLabels.append(row[0])
    for idx, labelName in enumerate(imgLabels):
        if not os.path.isdir(os.path.join(torchDir,labelName)):
            os.mkdir(os.path.join(torchDir,labelName))
    for idx in range(len(imgPaths)):
        shutil.copy( os.path.join(sourceDir,'image_{0:0>5d}.jpg'.format(imgNums[idx])), os.path.join(torchDir,imgLabels[idx],'image_{0:0>5d}.jpg'.format(imgNums[idx])) )

#ReShapeDataSet()

def DelDiffNameFiles():
    paths = glob.glob(r"F:\HC\HC\all\*.jpg")
    names = [ name.split(os.sep)[-1].split('.')[0] for name in paths]
    sumFile, sumDel = len(names), 0

    for idx, name in enumerate(names):
        if (name+'_1') in names:
            toDelPath = os.path.join(r"F:\HC\HC\all", name+'_1.jpg')
            mainPath = os.path.join(r"F:\HC\HC\all", name+'.jpg')
            if os.path.getsize(toDelPath) == os.path.getsize(mainPath):
                os.remove(toDelPath)
                sumDel+=1
        if idx % 25 == 0:
            print(f'\r {idx:d}/{sumFile:d}')

    print(f'\nsumDelFile = {sumDel:d}')
    
def BatchFileReName(root:str, srcSub:str, distSub:str):
    if not os.path.isdir(root):
        assert 0, "not os.path.isdir(root):" + root
    names = os.listdir(root)
    newNames = [n.replace(srcSub, distSub) for n in names]
    for idx, n in enumerate(names):
        if n != newNames[idx]:
            os.rename(os.path.join(root, n),
                    os.path.join(root, newNames[idx]))