from torch.utils.data import Dataset, DataLoader
import wfdb 
import numpy as np 
import pandas as pd 
import torch
import os 
from tqdm import tqdm


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000 
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig

class ECG_Dataset(Dataset): 
    def __init__(self, phase, data_dir, leads, raw_dir, fold): 
        super(ECG_Dataset, self).__init__()
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        df = pd.read_json(data_dir)
        folds = np.zeros(len(df))
        n = len(df)
        for i in range(10):
            start = int(n * i / 10)
            end = int(n * (i + 1) / 10)
            folds[start:end] = i + 1
        df['fold'] = np.random.permutation(folds)
        self.df = df[df['fold'].isin(fold)]
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else: 
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.n_leads = len(self.leads)
        self.labels = self.df['labels']
        self.phase = phase 

    def __getitem__(self, index):
        patient_id = self.df.iloc[index]['patient_id']
        data, _ = wfdb.rdsamp(os.path.join(self.raw_dir, patient_id))
        data  = transform(data, train = True)
        data = torch.Tensor(data)
        step = data.shape[0]
        data = data[-15000:, self.use_leads]
        result = torch.zeros((15000, self.n_leads))
        result[-step:, :] = data
        label = self.df.iloc[index]['labels']
        return result.T, torch.Tensor(label)
    def __len__(self):
        return(len(self.labels))

data = ECG_Dataset('train','formatted/thnig/formatted_files.json', 'all', 'data/training/Training_WFDB', [1,2])
print(data.__getitem__(0))

