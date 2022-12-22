'''
    @author Vinod Kumar Chauhan
    Institute of Biomedical Engineering,
    Department of Engineering,
    University of Oxford UK
    Last updated Dec. 22, 2022.
'''
import warnings
warnings.filterwarnings('ignore')

import wandb
import os
import sys
# import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import random
from random import SystemRandom
import torch
import torch.nn as nn

import src.utils as utils
from src.load_dataset import *
from src.coper_model import COPER
from src.lstm_model import LSTM_MODEL
from src.transformer_model import TRANSFORMER

############################################
############### COMMON SETUP ###############
args = utils.init_args()
start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
random.seed(args.random_seed)
################
num_epochs = args.niters
n_labels = args.num_labels
batch_size = args.batch_size
patience = args.patience
val_batch=64 # to avoid having only one class in evaluation
#47, 48 # 18, 24 #76, 48 #12, 1000
input_size = 76 if args.dataset == "mimic" else 47
seq_len = 48    # 48-0.8530; 10-0.8446
# args.num_latents = args.seq_len = seq_len # also need to turn on mask if output is continuous and num_latents=seq_len
args.seq_len = seq_len # also need to turn on mask if output is continuous and num_latents=seq_len
# args.num_latents = 48
############################################
# initilising wandb
# wandb.init(project=args.project, entity="abc")
wandb.init(mode="disabled")
wandb.run.name = args.model_type + "-" +str(args.drop if args.drop is not None else '0.00') + "-" + str(args.num_latents)
wandb.config = vars(args)
utils.makedirs(args.save)
experimentID = args.load
if experimentID is None:
    experimentID = int(SystemRandom().random()*100000)
# checkpoint
ckpt_path = os.path.join('./results/checkpoints/', args.model_type + '-'+ str(args.dataset) + '-F'+ str(args.fold) + "_D" +str(args.drop if args.drop is not None else '0') + "_S" + str(args.random_seed) + '.ckpt')
utils.makedirs('./results/checkpoints/')
# set logger
log_path = os.path.join("./results/logs/" + "exp_" + str(experimentID) + ".log")
utils.makedirs("./results/logs/")
logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), displaying=True)
logger.info("Experiment " + str(experimentID))
logger.info('args:\n')
logger.info(args)
logger.info(sys.argv)

######################################################
############### Prepare model and data ###############
carry_fwrd = args.model_type == "LSTM" or args.model_type == "PERCEIVER"
if args.dataset == "mimic":
    train_loader, test_loader, val_loader = load_mimic_loader(device, batch_size, val_batch, carry_fwrd, args.drop, sampler=False)
else:
    train_loader, test_loader, val_loader = load_physionet_nfold(args.fold, args.kfold, device, batch_size, carry_fwrd, args.drop, sampler=False)

# create model
if args.model_type == "COPER" or args.model_type == "PERCEIVER":
    model = COPER(args, n_labels, input_size, args.num_latents, args.latent_dim, args.rec_layers, args.units, nn.Tanh,
                     args.cont_in, args.cont_out, emb_dim=args.emb_dim, device=device).to(device)
elif args.model_type == "TRANSFORMER":
    model = TRANSFORMER(args, n_labels, input_size, args.num_latents, args.latent_dim, args.rec_layers, args.units, nn.Tanh,
                     args.cont_in, args.cont_out, emb_dim=args.emb_dim, device=device).to(device)                
elif args.model_type == "LSTM":
    model = LSTM_MODEL(n_labels, input_size, args.num_layers, args.bidirectional, args.latent_dim, args.lstm_dropout, device=device).to(device)
else:
    logger.error('Choose correct model.')
    raise Exception('Wrong model selected.')

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

# loss function
crit = nn.BCELoss()
crit_mse = nn.MSELoss()

# print model architecture and track gradients usin wandb
logger.info(model)
wandb.watch(model)

########################################
############### TRAINING LOOP ###############
best_auroc = 0
early_stopping = utils.EarlyStopping(patience=patience, path=ckpt_path, verbose=True, logger=logger)

num_batches = len(train_loader)
for epoch in range(1, num_epochs+1):
    model.train()
    train_loss = 0

    for batch_idx, data_dict in enumerate(train_loader):
        X, y = data_dict["X"], data_dict["y"]
        if args.drop is None:
            objd_t, pred_t, time_steps = data_dict["tp"], data_dict["tp"], data_dict["tp"]
            mask = None
        elif args.mask is False:
            objd_t, pred_t, time_steps = data_dict["objd_t"], data_dict["pred_t"], data_dict["tp"]
            mask = None
        else:
            objd_t, pred_t, time_steps, mask = data_dict["objd_t"], data_dict["pred_t"], data_dict["tp"], data_dict["mask"]
        opt.zero_grad()

        # forward pass
        output = model(X, time_steps, objd_t, pred_t)
        output = output.squeeze()

        loss = crit(output, y)

        # backward pass: calculate gradient and update weights
        loss.backward()
        opt.step()
        train_loss = train_loss + loss.item()

    logger.info('Train Loss:{:.6f}'.format(train_loss/len(train_loader)))
    wandb.log({"Train Loss": train_loss/len(train_loader)})
    model.eval()
    valid_auc = utils.evaluate(args, device, 'Val', val_loader, model, crit, logger, wandb, epoch, n_traj_samples=1)
    early_stopping(valid_auc, model)

    if early_stopping.early_stop:
        logger.info("Early stopping....")
        break

# create model 
if args.model_type == "COPER" or args.model_type == "PERCEIVER":
    best_model = COPER(args, n_labels, input_size, args.num_latents, args.latent_dim, args.rec_layers, args.units, nn.Tanh,
                     args.cont_in, args.cont_out, emb_dim=args.emb_dim, device=device).to(device)
elif args.model_type == "TRANSFORMER":
    best_model = TRANSFORMER(args, n_labels, input_size, args.num_latents, args.latent_dim, args.rec_layers, args.units, nn.Tanh,
                     args.cont_in, args.cont_out, emb_dim=args.emb_dim, device=device).to(device)                    
elif args.model_type == "LSTM":
    best_model = LSTM_MODEL(n_labels, input_size, args.num_layers, args.bidirectional, args.latent_dim, args.lstm_dropout, device=device).to(device)

best_model.load_state_dict(torch.load(ckpt_path))

# evaluation on test set
best_model.eval()
args.setting = 'Test' # this will retain the observed time-steps and generate the missing (applicable to continuous models)
utils.evaluate(args, device, 'Test-OG', test_loader, best_model, crit, logger, wandb, -1, n_traj_samples=1)
args.setting = '-1' # generate all time-steps
utils.evaluate(args, device, 'Test-G', test_loader, best_model, crit, logger, wandb, -1, n_traj_samples=1)
logger.info('...........Experiment ended.............')
#########################################################
