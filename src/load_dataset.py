from dataclasses import replace
import torch

import pickle
import json
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels, carry_frd, drop=None, mask=None):
        'Initialization'
        # print('Data:', data.shape, labels.shape)
        self.labels = labels
        self.data = data
        self.mask = mask
        self.drop = drop
        self.carry_frd = carry_frd
        self.time_steps = torch.tensor(np.array([i for i in range(data.shape[1])])/float(data.shape[1]-1)).to(data.device)
        # print('time steps size:', self.time_steps.shape)
        self.remove_t, self.objd_t = self.time_steps, self.time_steps
        if drop is not None:
            # self.drop_cforward_rand()
            self.drop_cforward_3p()

  def __len__(self):
        'Denotes the total number of samples'
        # print('Total samples:', self.labels.shape[0])
        return self.labels.shape[0]

  def __getitem__(self, index):
      if self.mask is None:
          mask = None
      else:
          mask = self.mask[index,:,:]
      x = self.data[index,:, :]
      if len(self.labels.shape)==1:
            y = self.labels[index]
      else:
            y = self.labels[index,:]

      if self.drop is None:
          data_dict ={ "X":x, "y":y, "tp":self.time_steps}
      elif self.mask is None:
          data_dict = { "X":x, "y":y, "tp":self.time_steps, "objd_t":self.objd_t, "pred_t":self.remove_t}
      else:
          data_dict = { "X":x, "y":y, "tp":self.time_steps, "objd_t":self.objd_t, "pred_t":self.remove_t, "mask":mask}

      return data_dict

  def drop_cforward_3p(self):
    '''
        This function will drop given percentage of time steps to generate irregular time-series.
        The time-steps will be replaced with carry forward technique.
    '''
    device = self.data.device
    T = self.data.shape[1]
    n = int(self.drop*T)
    m = int(n/3)

    remove_t = [i for i in range(T) if (i>0 and i<=m) or (i>=int(T/2-m/2) and i<int(T/2+m/2)) or i >= (T-n+2*m)]
    remove_t = np.sort(np.array(remove_t))
    objd_t = np.sort(np.delete(range(T), remove_t))
    # print(remove_t, objd_t)

    # use carryforward for baselines
    if self.carry_frd:
        print('using carry forward for missing steps')
        self.data[:,1:(m+1),:] = self.data[:,0,:].reshape(self.data.shape[0], 1, self.data.shape[2])
        self.data[:,int(T/2-m/2):int(T/2+m/2),:] = self.data[:,int(T/2-m/2)-1,:].reshape(self.data.shape[0], 1, self.data.shape[2])
        self.data[:,-(n-2*m):,:] = self.data[:,-(n-2*m)-1,:].reshape(self.data.shape[0], 1, self.data.shape[2])
    else:
        print('NOT using carry forward for missing steps')
        self.data = self.data[:,objd_t,:]

    remove_t = torch.tensor(remove_t / float(T-1)).to(device)
    remove_t, _ = torch.sort(remove_t)
    objd_t = torch.tensor(objd_t / float(T-1)).to(device)
    objd_t, _ = torch.sort(objd_t)

    self.remove_t, self.objd_t = remove_t, objd_t

    print('data: ', self.data.shape)

  def drop_cforward_rand(self):
    device = self.data.device
    T = self.data.shape[1]
    n = int(self.drop*T)
    assert n < T, 'percentage drop should be less than 1.0'
    remove_t = random.sample(range(1, T-1), n)
    remove_t = np.sort(np.array(remove_t))
    objd_t = np.sort(np.delete(range(T), remove_t))

    # find time steps for replacing with
    replace_with_t = np.zeros_like(remove_t)
    for i, t in enumerate(remove_t):
        if sum(t < objd_t)>0:
            for j in range(t-1,-1,-1):
                if j in objd_t:
                    replace_with_t[i] = j
                    break
        else:
            for j in range(t+1,T):
                if j in objd_t:
                    replace_with_t[i] = j
                    break

    # remove time steps randomly
    if self.carry_frd:
        self.data[:,remove_t,:] = self.data[:,replace_with_t,:]
    else:
        self.data = self.data[:,objd_t,:]
    # print(remove_t, replace_with_t, 'JMD--------------------')
    remove_t = torch.tensor(remove_t / float(T-1)).to(device)
    remove_t, _ = torch.sort(remove_t)
    objd_t = torch.tensor(objd_t / float(T-1)).to(device)
    objd_t, _ = torch.sort(objd_t)

    self.remove_t, self.objd_t = remove_t, objd_t


def load_mimic_loader(device, batch, carry_frd, drop, sampler):
    with open('paths.json', 'r') as f:
        file_name = json.load(f)["mimic3_mortality"]
    with open(file_name,'rb') as fp:
        details, X_train, y_train, X_val, y_val, X_test, y_test, _ = pickle.load(fp)

    # print(X_train.shape, y_train.shape, '------------------')
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    X_val = torch.Tensor(X_val).to(device)
    y_val = torch.Tensor(y_val).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    training_set = Dataset(X_train, y_train, carry_frd, drop=drop)
    test_set = Dataset(X_test, y_test, carry_frd, drop=drop)
    val_set = Dataset(X_val, y_val, carry_frd, drop=drop)

    if sampler:
        # deal with imbalanced dataset - weighted sampling
        class_sample_count = np.unique(training_set.labels.cpu().data.numpy(), return_counts=True)[1]
        weight = 2. / class_sample_count
        print('class weights:', weight)
        samples_weight = weight[training_set.labels.cpu().data.numpy().astype('int')]
        samples_weight = torch.from_numpy(samples_weight).float().to(device)

        sampler = WeightedRandomSampler(samples_weight, num_samples=
                                        len(samples_weight), replacement=True)
        # print('JMD: using weightedsampler......')
        train_loader = DataLoader(training_set, batch_size=batch, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True) #, collate_fn=collate_batch

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=False)

    return train_loader, test_loader, val_loader


def collate_batch(batch):

    X_list, y_list = [], []

    for (_X,_y) in batch:
        print(_X.shape)
        y_list.append(_y)
        X_list.append(_X)

    y_list = torch.stack(y_list, dim=0).to(batch[0][0].device)

    X_list = torch.nn.utils.rnn.pad_sequence(X_list, batch_first=True, padding_value=0)

    # print(X_list,y_list)

    return X_list,y_list,