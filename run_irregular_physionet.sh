#!/bin/bash
#SBATCH --time=1
eval "$(conda shell.bash hook)"
conda activate base

####################################
###############################################################################################################
###############################################################################################################
data="physionet" # physionet, mimic
for seed in 1 2 3 4 5 ; do
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1 --drop 0.25
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1 --drop 0.50
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1 --drop 0.75
    
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2 --drop 0.25
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2 --drop 0.50
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2 --drop 0.75
    
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3 --drop 0.25
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3 --drop 0.50
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3 --drop 0.75
    
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4 --drop 0.25
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4 --drop 0.50
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4 --drop 0.75
    
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5 --drop 0.25
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5 --drop 0.50
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5 --drop 0.75
    
    ####################################################
    # # Perceiver
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 1 --project physionet-irregular-A --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 1 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 1 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 1 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.75
    
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 2 --project physionet-irregular-A --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 2 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 2 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 2 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.75
    
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 3 --project physionet-irregular-A --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 3 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 3 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 3 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.75
    
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 4 --project physionet-irregular-A --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 4 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 4 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 4 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.75
    
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 5 --project physionet-irregular-A --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 5 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 5 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 5 --project physionet-irregular-A --random-seed $seed --num-latents 48 --drop 0.75
    
    #######################################
    # TRANSFORMER
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 1 --project physionet-irregular --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 1 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 1 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 1 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.75

    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 2 --project physionet-irregular --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 2 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 2 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 2 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.75
    
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 3 --project physionet-irregular --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 3 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 3 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 3 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.75
    
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 4 --project physionet-irregular --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 4 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 4 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 4 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.75
    
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 5 --project physionet-irregular --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 5 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 5 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --fold 5 --project physionet-irregular --random-seed $seed --num-latents 48 --drop 0.75

    # #######################################
    # # mTAND
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1 --drop 0.25
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1 --drop 0.5
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1 --drop 0.75
    
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2 --drop 0.25
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2 --drop 0.5
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2 --drop 0.75
    
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3 --drop 0.25
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3 --drop 0.5
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3 --drop 0.75
    
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4 --drop 0.25
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4 --drop 0.5
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4 --drop 0.75
    
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5 --drop 0.25
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5 --drop 0.5
    # python run_exp.py --model-type 'mTAND' --alpha 100 --niters 300 --lr 0.0001 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5 --drop 0.75
    
    ######################################
    # COPER
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 1
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 1 --drop 0.25
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 1 --drop 0.50
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 1 --drop 0.75
    
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 2
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 2 --drop 0.25
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 2 --drop 0.50
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 2 --drop 0.75
    
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 3
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 3 --drop 0.25
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 3 --drop 0.50
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 3 --drop 0.75
    
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 4
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 4 --drop 0.25
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 4 --drop 0.50
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 4 --drop 0.75
    
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 5
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 5 --drop 0.25
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 5 --drop 0.50
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project physionet-irregular-A --random-seed $seed --num-latents 48 --fold 5 --drop 0.75
    
    # ###################################
    # # # LODE
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1 --drop 0.25
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1 --drop 0.50
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 1 --drop 0.75
    
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2 --drop 0.25
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2 --drop 0.5
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 2 --drop 0.75
    
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3 --drop 0.25
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3 --drop 0.5
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 3 --drop 0.75

    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4 --drop 0.25
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4 --drop 0.5
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 4 --drop 0.75

    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5 --drop 0.25
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5 --drop 0.5
    # python run_exp.py --model-type 'LODE'  --niters 300 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project physionet-irregular-A --random-seed $seed --fold 5 --drop 0.75

done

###############################################################################################################

conda deactivate
