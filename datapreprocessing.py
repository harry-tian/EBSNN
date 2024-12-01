import random
from time import time
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm

import os, sys
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.join(base_path),os.path.join(base_path, "../"),base_path.rsplit('/')[0]])

def segment_array(arr, segment_len=8): 
    # output: L * N
    result = []
    for subarray in arr:
        for i in range(0, len(subarray), segment_len):
            segment = subarray[i:i+segment_len]
            if len(segment) < segment_len:
                segment.extend([0]*(segment_len-len(segment)))
            result.append(segment)
    return result


def calculate_alpha(counter, mode='normal'):
    if mode == 'normal':
        alpha = torch.tensor(counter, dtype=torch.float32)
        alpha = alpha / alpha.sum(0).expand_as(alpha)
    elif mode == 'invert':
        alpha = torch.tensor(counter, dtype=torch.float32)
        alpha_sum = alpha.sum(0)
        alpha_sum_expand = alpha_sum.expand_as(alpha)
        alpha = (alpha_sum - alpha) / alpha_sum_expand
    # fill all zeros to ones
    alpha[alpha==0.] = 1.
    return alpha

class Loader():
    def __init__(self, X, Y, label_dict, batch_size, segment_len=8):
        self.X = []
        self.Y = Y
            
        self.segment_len = segment_len
        self.batch_size = batch_size

        self.alpha = Counter(self.Y)
        for i in label_dict.values():
            if i not in self.alpha:
                self.alpha[i] = 0
        # TODO(DCMMC): consistent with FlowLoader?
        counter = [self.alpha[k] for k in sorted(self.alpha.keys())]
        self.alpha = calculate_alpha(counter, mode='invert')
        
        print(f"Segmenting {len(X)} packets with segment_len={self.segment_len}")
        for packet in tqdm(X):
            packet_segmented = torch.tensor(segment_array(packet, self.segment_len)).long() # (L, N) 
            self.X.append(packet_segmented)


    def __len__(self):
        return int(np.ceil(len(self.Y) / self.batch_size))

    def __getitem__(self, idx):
        batch_X, batch_y = [], []
        
        for i in range(idx * self.batch_size, min(len(self.Y), (idx+1) * self.batch_size)):
            batch_X.append(self.X[i])
            batch_y.append(self.Y[i])
        
        batch_y = torch.tensor(batch_y, dtype=torch.long)

        return (batch_X, batch_y)


def get_dataloader(dataset_dir, batch_size, data_split=[0.6, 0.2, 0.2], segment_len=8):
    train_size, val_size, test_size = data_split
    X, Y = [], []
    labels = [f[:-4] for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
    label_dict = {d:i for i,d in enumerate(labels)} # class: idx
    for file in os.listdir(dataset_dir):  # subdir level
        if file.endswith('.pkl'):
            label = file[:-4]
            y = label_dict[label]
            file_name = os.path.join(dataset_dir, file)
            with open(file_name, 'rb') as f:
                try:
                    while True:
                        packet = pickle.load(f)
                        X.append(packet)
                        Y.append(y)
                except EOFError:
                    print("Finished reading " + file_name)
    ## packet format: [[IP header], [TCP/UDP header], [payload]]
    # all stored as list of 8-bit ints
    # payload may be empty

    all_idx = list(range(len(X)))

    train_idx, nontrain_idx, _, _ = train_test_split(all_idx, [0 for _ in all_idx], test_size=val_size+test_size, random_state=0)
    X_train = [X[i] for i in train_idx]
    Y_train = [Y[i] for i in train_idx]

    val_idx, test_idx, _, _ = train_test_split(nontrain_idx, [0 for _ in nontrain_idx], test_size=test_size/(val_size+test_size), random_state=0)
    X_test = [X[i] for i in test_idx]
    Y_test = [Y[i] for i in test_idx]
    X_val = [X[i] for i in val_idx]
    Y_val = [Y[i] for i in val_idx]

    print(f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    train_loader = Loader(X_train, Y_train, label_dict, batch_size, segment_len=segment_len )
    val_loader = Loader(X_val, Y_val, label_dict, batch_size, segment_len=segment_len )
    test_loader = Loader(X_test, Y_test, label_dict, batch_size, segment_len=segment_len )
        
    return train_loader, val_loader, test_loader, label_dict

def get_testloader(dataset_dir, batch_size, segment_len=8):
    X, Y = [], []
    labels = [f[:-4] for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
    label_dict = {d:i for i,d in enumerate(labels)} # class: idx
    for file in os.listdir(dataset_dir):  # subdir level
        if file.endswith('.pkl'):
            label = file[:-4]
            y = label_dict[label]
            file_name = os.path.join(dataset_dir, file)
            with open(file_name, 'rb') as f:
                try:
                    while True:
                        packet = pickle.load(f)
                        X.append(packet)
                        Y.append(y)
                except EOFError:
                    print("Finished reading " + file_name)
    ## packet format: [[IP header], [TCP/UDP header], [payload]]
    # all stored as list of 8-bit ints
    # payload may be empty
        
    return Loader(X, Y, label_dict, batch_size, segment_len=segment_len), label_dict

################################################
# DCMMC: new getloader for flow classification #
################################################


class FlowLoader():
    def __init__(self, X, Y, batch_size, segment_len=8):
        self.X = []
        self.Y = []
            
        self.segment_len = segment_len
        self.batch_size = batch_size
        
        for flow,y in zip(X, Y):
            for packet in flow:
                self.X.append(torch.tensor(segment_array(packet, self.segment_len)).long())
                self.Y.append(y)

    def __len__(self):
        return int(np.ceil(len(self.Y) / self.batch_size))

    def __getitem__(self, idx):
        batch_X, batch_y = [], []
        
        for i in range(idx * self.batch_size, min(len(self.Y), (idx+1) * self.batch_size)):
            batch_X.append(self.X[i])
            batch_y.append(self.Y[i])
        
        return (batch_X, batch_y)


def get_dataloader_flow(dataset_dir, first_k_packets=3, segment_len=8):
    batch_size = first_k_packets
    X, Y = [], []
    labels = [f[:-4] for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
    label_dict = {d:i for i,d in enumerate(labels)} # class: idx
    for file in os.listdir(dataset_dir):  # subdir level
        if file.endswith('.pkl'):
            label = file[:-4]
            y = label_dict[label]
            file_name = os.path.join(dataset_dir, file)
            with open(file_name, 'rb') as f:
                flows = pickle.load(f)
            
            if len(flows) > 0:
                for flow in flows:
                    X.append(flow[:first_k_packets])
                    Y.append(y)
            else:
                X.append(flow[:first_k_packets])
                Y.append(y)


    return FlowLoader(X, Y, batch_size, segment_len), label_dict

