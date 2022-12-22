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
            self.drop_cforward_rand()
            # self.drop_cforward_3p()

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
    print(remove_t, objd_t)

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

def load_physionet_anshul_loader(device, batch, carry_frd, drop, sampler):
    from sklearn.model_selection import StratifiedKFold

    with open('paths.json', 'r') as f:
        file_names = json.load(f)
        X = np.load(file_names["physionet_2012_data"])
        y = np.load(file_names["physionet_2012_labels"])

    print('data:', X.shape, y.shape)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5)

    # # scale the data
    # # calc min/max values
    # mins = np.zeros(X_train.shape[-1])
    # maxs = np.zeros(X_train.shape[-1])
    # for i in range(len(mins)):
    #     mins[i] = X_train[:,:,i].reshape(-1).min()
    #     maxs[i] = X_train[:,:,i].reshape(-1).max()
    #     # mins[i] = X_train[:,:,i].reshape(-1).mean()
    #     # maxs[i] = X_train[:,:,i].reshape(-1).std()

    # # scaling the values
    # for i in range(len(mins)):
    #     X_train[:,:,i] = (X_train[:,:,i] - mins[i])/(maxs[i] - mins[i])
    #     X_test[:,:,i] = (X_test[:,:,i] - mins[i])/(maxs[i] - mins[i])
    #     # X_train[:,:,i] = (X_train[:,:,i] - mins[i])/(maxs[i])
    #     # X_test[:,:,i] = (X_test[:,:,i] - mins[i])/(maxs[i])

    # convert to Tensor
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    # X_val = torch.Tensor(X_val).to(device)
    # y_val = torch.Tensor(y_val).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    training_set = Dataset(X_train, y_train, carry_frd, drop=drop)
    test_set = Dataset(X_test, y_test, carry_frd, drop=drop)
    # val_set = Dataset(X_val, y_val)

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
    val_loader = test_loader # torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=False)

    return train_loader, test_loader, val_loader

def load_physionet_nfold(fold, K, device, batch, carry_frd, drop, sampler):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    with open('paths.json', 'r') as f:
        file_names = json.load(f)
        X = np.load(file_names["physionet_2012_data"])
        y = np.load(file_names["physionet_2012_labels"])

    print('total data points:', X.shape, y.shape)

    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=2022)
    kf.get_n_splits(X,y)
    i=0
    for train_index, test_index in kf.split(X,y):
        i = i+1
        if fold==i:
            print('Fold: ', i)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)

    scale = NormaliseTimeSeries()
    X_train = scale.fit(X_train)
    X_val = scale.normalise(X_val)
    X_test = scale.normalise(X_test)
    print('Data:', X_train.shape, X_val.shape, X_test.shape)

    # convert to Tensor
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

# def load_physionet_anshul_loader(device, batch=32, sampler=False):
#     from sklearn.model_selection import StratifiedKFold

#     with open('paths.json', 'r') as f:
#         file_names = json.load(f)
#         X = np.load(file_names["physionet_2012_data"])
#         y = np.load(file_names["physionet_2012_labels"])

#     print('data:', X.shape, y.shape)
#     #for train_idxs, test_idxs in cv.split(X, y, groups):
#     kf = StratifiedKFold(n_splits=2)
#     kf.get_n_splits(X,y)
#     # y_test_all=[]
#     # y_pred_all=[]

#     for train_index, test_index in kf.split(X,y):

#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#     # convert to Tensor
#     X_train = torch.Tensor(X_train).to(device)
#     y_train = torch.Tensor(y_train).to(device)
#     # X_val = torch.Tensor(X_val).to(device)
#     # y_val = torch.Tensor(y_val).to(device)
#     X_test = torch.Tensor(X_test).to(device)
#     y_test = torch.Tensor(y_test).to(device)

#     training_set = Dataset(X_train, y_train)
#     test_set = Dataset(X_test, y_test)
#     # val_set = Dataset(X_val, y_val)

#     if sampler:
#         # deal with imbalanced dataset - weighted sampling
#         class_sample_count = np.unique(training_set.labels.cpu().data.numpy(), return_counts=True)[1]
#         weight = 2. / class_sample_count
#         print('class weights:', weight)
#         samples_weight = weight[training_set.labels.cpu().data.numpy().astype('int')]
#         samples_weight = torch.from_numpy(samples_weight).float().to(device)

#         sampler = WeightedRandomSampler(samples_weight, num_samples=
#                                         len(samples_weight), replacement=True)
#         # print('JMD: using weightedsampler......')
#         train_loader = DataLoader(training_set, batch_size=batch, sampler=sampler)
#     else:
#         train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True) #, collate_fn=collate_batch

#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, shuffle=False)
#     val_loader = test_loader # torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=False)

#     return train_loader, test_loader, val_loader


def load_mimic_loader(device, batch, val_batch, carry_frd, drop, sampler):
    # with open('paths.json', 'r') as f:
    #     file_names = json.load(f)
    #     train_file = file_names["mimic3_train_path"]
    #     test_file = file_names["mimic3_test_path"]
    #     val_file = file_names["mimic3_val_path"]

    #     with open(train_file, 'rb') as f:
    #         X_train, y_train = pickle.load(f)
    #     with open(test_file, 'rb') as f:
    #         X_test, y_test = pickle.load(f)
    #     with open(val_file, 'rb') as f:
    #         X_val, y_val = pickle.load(f)
    # print('X_train.shape, ', X_test)

    with open('paths.json', 'r') as f:
        file_name = json.load(f)["mimic3_mortality"]
    # load data /data/Vinod/Data/datasets/mimic-iii-clinical-database-1.4/4_ready_for_model
    # with open("/Users/vinodkumar/DATA/Oxford/mimi3_ihm/mortality.data",'rb') as fp:
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

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=val_batch, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch, shuffle=False)

    return train_loader, test_loader, val_loader


def load_pcb_data(device):
    # load data
    with open("/Users/vinodkumar/DATA/CODE/datasets/PTBXL/PTBXL_form_ptb_data.db",'rb') as fp:
        X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(fp)

        # print(type(X_train))
        # print(X_train[0,0,:])
    # # scale data
    # ss = StandardScaler()
    # X_train = ss.fit_transform(X_train)
    # X_val = ss.transform(X_val)
    # X_test = ss.transform(X_test)

    # convert to Tensor
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    X_val = torch.Tensor(X_val).to(device)
    y_val = torch.Tensor(y_val).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)


    training_set = Dataset(X_train, y_train)
    test_set = Dataset(X_test, y_test)
    val_set = Dataset(X_val, y_val)

    return training_set, test_set, val_set


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

class NormaliseTimeSeries():
    def __init__(self):
        self.mins = None
        self.maxs = None
        self.eps = 1e-9

    def fit(self, X_train):
        self.mins = np.zeros(X_train.shape[-1])
        self.maxs = np.zeros(X_train.shape[-1])
        for i in range(len(self.mins)):
            # self.mins[i] = X_train[:,:,i].reshape(-1).min()
            # self.maxs[i] = X_train[:,:,i].reshape(-1).max()
            self.mins[i] = X_train[:,:,i].reshape(-1).mean()
            self.maxs[i] = X_train[:,:,i].reshape(-1).std() + self.eps
        # scaling the values
        for i in range(len(self.mins)):
            # X_train[:,:,i] = (X_train[:,:,i] - self.mins[i])/(self.maxs[i] - self.mins[i])
            X_train[:,:,i] = (X_train[:,:,i] - self.mins[i])/(self.maxs[i])
        return X_train

    def normalise(self, X):
        assert self.mins is not None, 'First call fit function.'
        # scaling the values
        for i in range(len(self.mins)):
            # X[:,:,i] = (X[:,:,i] - self.mins[i])/(self.maxs[i] - self.mins[i])
            X[:,:,i] = (X[:,:,i] - self.mins[i])/(self.maxs[i])
        return X

if __name__ == '__main__':
    torch.manual_seed(1991)

    device = torch.device("cpu")

    train_loader, test_loader, val_loader = load_mimic_loader(device, 32, None)
