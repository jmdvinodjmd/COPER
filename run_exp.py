'''
    @author Vinod Kumar Chauhan
    Institute of Biomedical Engineering,
    Department of Engineering,
    University of Oxford UK
'''
import warnings
warnings.filterwarnings('ignore')

import wandb
import os
import sys
import time
import argparse
import numpy as np
import random
from einops import rearrange, repeat
from random import SystemRandom
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import src.utils as utils
from src.load_dataset import *
from src.coper_model import COPER
from src.lstm_model import LSTM_MODEL

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

input_size = 76
seq_len = 48
args.num_latents = args.seq_len = seq_len

############################################
# initilising wandb
# wandb.init(project="..", entity="..")
wandb.init(mode="disabled")
wandb.run.name = args.model_type + "-" +str(args.drop if args.drop is not None else '0.00')
wandb.config = vars(args)
utils.makedirs(args.save)
experimentID = args.load
if experimentID is None:
    experimentID = int(SystemRandom().random()*100000)
# checkpoint
ckpt_path = os.path.join('./results/checkpoints/', args.model_type+ "_" +str(args.drop if args.drop is not None else '0') + "_" + str(experimentID) + '.ckpt')
utils.makedirs('./results/checkpoints/')
# set logger
log_path = os.path.join("./results/logs/" + "exp_" + str(experimentID) + ".log")
utils.makedirs("./results/logs/")
logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
logger.info("Experiment " + str(experimentID))
logger.info('args:\n')
logger.info(args)
logger.info(sys.argv)

######################################################
############### Prepare model and data ###############
carry_fwrd = args.model_type == "LSTM" or args.model_type == "PERCEIVER"
train_loader, test_loader, val_loader = load_mimic_loader(device, batch_size, carry_fwrd, args.drop, sampler=False)

# create model
if args.model_type == "COPER" or args.model_type == "PERCEIVER":
    model = COPER(args, n_labels, input_size, args.num_latents, args.latent_dim, args.rec_layers, args.units, nn.Tanh,
                     args.cont_in, args.cont_out, emb_dim=args.emb_dim, device=device).to(device)
elif args.model_type == "LSTM":
    model = LSTM_MODEL(n_labels, input_size, args.num_layers, args.bidirectional, args.latent_dim, args.lstm_dropout, device=device).to(device)
else:
    logger.error('Choose correct model.')
    raise Exception('Wrong model selected.')

opt = torch.optim.Adam(model.parameters(), lr=args.lr) #, weight_decay=1e-5 for l2-regularisation

# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5, verbose=True)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs, verbose=True)

# loss function
crit = nn.BCELoss()

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
        # print(output, y)
        loss = crit(output, y)

        # backward pass: calculate gradient and update weights
        loss.backward()
        opt.step()
        train_loss = train_loss + loss.item()

    # scheduler.step()
    logger.info('Train Loss:{:.6f}'.format(train_loss/len(train_loader)))
    wandb.log({"Train Loss": train_loss/len(train_loader)})
    model.eval()
    valid_auc = utils.evaluate(args, device, 'Val', val_loader, model, crit, logger, wandb, epoch, time_steps)
    early_stopping(valid_auc, model)

    if early_stopping.early_stop:
        logger.info("Early stopping....")
        break

# load the best model from early stopping
# create model
if args.model_type == "COPER" or args.model_type == "PERCEIVER":
    best_model = COPER(args, n_labels, input_size, args.num_latents, args.latent_dim, args.rec_layers, args.units, nn.Tanh,
                     args.cont_in, args.cont_out, emb_dim=args.emb_dim, device=device).to(device)
elif args.model_type == "LSTM":
    best_model = LSTM_MODEL(n_labels, input_size, args.num_layers, args.bidirectional, args.latent_dim, args.lstm_dropout, device=device).to(device)

best_model.load_state_dict(torch.load(ckpt_path))

# evaluation on test set
best_model.eval()
utils.evaluate(args, device, 'Test', test_loader, best_model, crit, logger, wandb, -1, time_steps)
logger.info('...........Experiment ended.............')
#########################################################