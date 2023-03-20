'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-21 00:12:32
LastEditors: BHM-Bob
LastEditTime: 2023-03-21 00:31:56
Description: 
'''
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from mbapy.dl_torch.utils import GlobalSettings, Mprint


def denarmalize_img(x, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    return x * std + mean

@torch.jit.script
class RandomSeqGenerator(object):
    def __init__(self, seqs:torch.Tensor, lables:torch.Tensor, args:GlobalSettings,
                 multiLoads:int = 6, maskRatio:float = 0.3, device:str = 'cpu'):
        self.args = args
        self.seqs = seqs
        self.lables = lables
        self.device = device
        self.sumSeqs = len(seqs)
        self.multiLoads = multiLoads
        self.isMask = maskRatio > 0
        self.maskRatio = torch.tensor(maskRatio, device = device)
    def __call__(self, idx:int):
        idxs = torch.randint(0, self.sumSeqs, size = [self.multiLoads])
        idxs[0] = idx
        seqs, lables = self.seqs[idxs], self.lables[idxs]
        if self.isMask:
            mask = torch.rand(self.args.input['single_shape'], device=self.device).ge(self.maskRatio)
            seqs.mul_(mask)
        return seqs, lables        

class SubDataSet(Dataset):
    def __init__(self, x:list, labels:list, args:GlobalSettings,
                 multiLoads:int = 6, maskRatio:float = 0.3, device:str = 'cpu'):
        super(SubDataSet,self).__init__()
        self.args = args
        self.multiLoads = multiLoads
        self.maskRatio = maskRatio
        self.multiLoads = multiLoads
        self.getRandomSeq = RandomSeqGenerator(torch.cat(x, dim = 0),
                                               torch.cat(labels, dim = 0),
                                               args, multiLoads, maskRatio, device)
        self.sumSeqs = len(x)
    def __getitem__(self, idx):
        # seqs : [M, L], labels: [CN,]
        return self.getRandomSeq(idx)
    def __len__(self):
        return self.sumSeqs

class DataSetRAM():
    def __init__(self, args:GlobalSettings, loadPart:str = 'pre', device:str = 'cpu'):
        super(DataSetRAM,self).__init__()
        self.args = args
        self.labelLen = args.sumClass
        self.x = []
        self.labels = []
        self.loadRatio = args.load_db_ratio
        self.multiLoads = args.multiLoads
        self.maskRatio = args.maskRatio
        self.batchSize =args.batchSize
        self.loadPart = loadPart
        self.device = device

    def GetSubDS(self, divideList, multiLoads = None, maskRatio = None):
        """divideList : [0, 0.7, 0.9, 1] => train_70% val_20% test_10%
        """
        ret = []
        for idx in range(len(divideList) - 1):
            index1 = int(divideList[idx  ]*self.sumData)
            index2 = int(divideList[idx+1]*self.sumData)
            mL = self.multiLoads if multiLoads is None else multiLoads[idx]
            mR = self.maskRatio if maskRatio is None else maskRatio[idx]
            ret.append(
                DataLoader(
                    SubDataSet(seqs = self.x[ index1 : index2],
                               labels = self.labels[ index1 : index2],
                               args = self.args,
                               multiLoads = mL,
                               maskRatio = mR,
                               device = self.device)
                    ,batch_size = self.batchSize, shuffle = True, drop_last=True))
            self.args.mp.mprint(f'dataSet{idx:d} len:{index2-index1:d} ML:{mL:d} Mask:{mR:4.2f}')
        return ret