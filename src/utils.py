import argparse
from asyncio.log import logger
import os
import logging
import torch
import numpy as np

import src.metrics as metrics



def init_args():
    parser = argparse.ArgumentParser('ODE')

    parser.add_argument('--UQ', type=int, default=2)
    parser.add_argument('--model-type', type=str, default='TCN', help="model to run: COPER, LSTM, mTAND, LODE, TCN...")
    parser.add_argument('--setting', type=str, default='Train', help="need to set to Test while testing the ODE to retain observed points.")
    parser.add_argument('--drop',  type=float, default=None, help="percentage of points to remove.")
    parser.add_argument('--niters', type=int, default=120)
    parser.add_argument('--num-labels', type=int, default=1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr',  type=float, default=1e-04, help="Starting learning rate.")
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--save', type=str, default='results/checkpoints/', help="Path for save checkpoints")
    parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
    parser.add_argument('-r', '--random-seed', type=int, default=2022, help="Random_seed")
    parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, activity, hopper, periodic")
    parser.add_argument('--project', type=str, default='perceiver', help="name of the project for wandb")
    parser.add_argument('--num-latents', type=int, default=48, help="number of latents")
    parser.add_argument('--rec-dims', type=int, default=40, help="Dimensionality of the recognition model (ODE or RNN).")
    parser.add_argument('--rec-layers', type=int, default=3, help="Number of layers in ODE func")
    parser.add_argument('--units', '-u', type=int, default=100, help="Number of units per layer in ODE func")
    parser.add_argument('--emb-dim', type=int, default=32, help="embedding dimension")
    parser.add_argument('--mask', action='store_true')
    
    parser.add_argument('--cont-in', action='store_true')
    parser.add_argument('--cont-out', action='store_true')
    parser.add_argument('--self-per-cross-attn', type=int, default=1, help="self_per_cross_attn")
    parser.add_argument('--latent-heads', type=int, default=1, help="latent_heads")
    parser.add_argument('--cross-heads', type=int, default=1, help="cross-heads")
    parser.add_argument('--cross-dim-head', type=int, default=8, help="cross_dim_head")
    parser.add_argument('--latent-dim-head', type=int, default=8, help=".")
    parser.add_argument('--latent-dim', type=int, default=8, help=".")
    parser.add_argument('--ff-dropout', type=float, default=0.5)
    parser.add_argument('--att-dropout', type=float, default=0.5)
    parser.add_argument('--ode-dropout', type=float, default=0.1)

    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--num-layers', type=int, default=2, help=".")
    parser.add_argument('--lstm-dropout', type=float, default=0.1)

    parser.add_argument('--rec-hidden', type=int, default=64)
    parser.add_argument('--gen-hidden', type=int, default=32)
    parser.add_argument('--enc-num-heads', type=int, default=1)
    parser.add_argument('--dec-num-heads', type=int, default=1)
    parser.add_argument('--embed-time', type=int, default=256)
    parser.add_argument('--learn-emb', action='store_true')
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--std', type=float, default=0.01)
    parser.add_argument('--k-iwae', type=int, default=1)
    parser.add_argument('--alpha', type=int, default=5.)

    parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
    parser.add_argument('--gen-layers', type=int, default=3, help="Number of layers in ODE func in generative ODE")
    parser.add_argument('-g', '--gru-units', type=int, default=50, help="Number of units per layer in each of GRU update networks")
    parser.add_argument('--linear-classif', action='store_false', help="If using a classifier, use a linear classifier instead of 1-layer NN")

    args = parser.parse_args()

    return args


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def evaluate(args, device, setting, test_loader, model, crit, logger, wandb, epoch, n_traj_samples=None):
    
    test_loss = 0
    predictions = torch.Tensor([]).to(device)
    labels = torch.Tensor([]).to(device)
    # with torch.no_grad():
    for data_dict in test_loader:
        X, y = data_dict["X"], data_dict["y"]
        if args.drop is None:
            objd_t, pred_t, time_steps = data_dict["tp"], data_dict["tp"], data_dict["tp"]
            mask = None
        elif args.mask is False:
            objd_t, pred_t, time_steps = data_dict["objd_t"], data_dict["pred_t"], data_dict["tp"]
            mask = None
        else:
            objd_t, pred_t, time_steps, mask = data_dict["objd_t"], data_dict["pred_t"], data_dict["tp"], data_dict["mask"]

        # forward pass
        output = model(X, time_steps, objd_t, pred_t)
        output = output.squeeze()
        predictions = torch.cat((predictions, output))
        labels = torch.cat((labels, y))
        test_loss +=crit(output, y).item()

    test_loss = test_loss / len(test_loader)
    # calculate metrics
    wandb.log({setting+" loss":test_loss})
    logger.info('{} Epoch: {:04d} | {} Loss {:.6f}'.format(setting, epoch, setting, test_loss))
    results = metrics.print_metrics_binary_classification(labels.data.cpu().numpy(),
                predictions.data.cpu().numpy(), setting, verbose=1, logger=logger, wandb=wandb)
    logger.info('\n')

    if 'Test' in setting:
        np.savez('./results/Predictions_'+args.model_type + '-' + setting + '-'+ str(args.dataset) + '-'+ str(args.fold) + '-'+ str(args.random_seed) + '-'+ str(args.drop), predictions.data.cpu().numpy())
        
    return results[setting+' AUROC']

def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()

def evaluate_uq(args, device, setting, test_loader, model, crit, logger, wandb, epoch, n_traj_samples=None):
    for i in range(args.UQ):
        test_loss = 0
        predictions = torch.Tensor([]).to(device)
        labels = torch.Tensor([]).to(device)
        
        enable_dropout(model)
        # with torch.no_grad()
        for data_dict in test_loader:
            X, y = data_dict["X"], data_dict["y"]
            if args.drop is None:
                objd_t, pred_t, time_steps = data_dict["tp"], data_dict["tp"], data_dict["tp"]
                mask = None
            elif args.mask is False:
                objd_t, pred_t, time_steps = data_dict["objd_t"], data_dict["pred_t"], data_dict["tp"]
                mask = None
            else:
                objd_t, pred_t, time_steps, mask = data_dict["objd_t"], data_dict["pred_t"], data_dict["tp"], data_dict["mask"]
            
            # forward pass
            output = model(X, time_steps, objd_t, pred_t)
            output = output.squeeze()
            # print('data size:', data.shape)
            predictions = torch.cat((predictions, output))
            labels = torch.cat((labels, y))
            test_loss +=crit(output, y).item()

        test_loss = test_loss / len(test_loader)
        # calculate metrics
        wandb.log({setting+" loss":test_loss})
        logger.info('{} Epoch: {:04d} | {} Loss {:.6f}'.format(setting, epoch, setting, test_loss))
        results = metrics.print_metrics_binary_classification(labels.data.cpu().numpy(),
                    predictions.data.cpu().numpy(), setting, verbose=1, logger=logger, wandb=wandb)
        
        if 'Test' in setting:
            np.savez('./results/Predictions_'+args.model_type+ '-UQ-'+ str(i)  + '-' + setting + '-'+ str(args.dataset) + '-'+ str(args.fold) + '-'+ str(args.random_seed) + '-'+ str(args.drop), preds=predictions.data.cpu().numpy(), labels=labels.data.cpu().numpy())

        logger.info('\n')
        
    return results[setting+' AUROC']

def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


# https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = np.Inf
        self.delta = delta
        self.path = path
        # self.trace_func = trace_func
        self.logger = logger

    def __call__(self, val_auc, model):

        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation acu increase.'''
        if self.verbose:
            self.logger.info(f'Validation auc increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_auc_max = val_auc
        
