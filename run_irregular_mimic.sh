#!/bin/bash
#SBATCH --time=1
eval "$(conda shell.bash hook)"
conda activate base

####################################
###############################################################################################################
###############################################################################################################
data="mimic" # physionet, mimic
for seed in 1 2 3 4 5 ; do
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --drop 0.25 
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --drop 0.50
    python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --drop 0.75

    # ######################################
    # # mTAND
    # python run_exp.py --model-type 'mTAND' --alpha 5 --niters 300 --lr 0.0001 --batch-size 128 --rec-hidden 256 --gen-hidden 50 --latent-dim 128 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project mimic-irregular-A --random-seed $seed
    # python run_exp.py --model-type 'mTAND' --alpha 5 --niters 300 --lr 0.0001 --batch-size 128 --rec-hidden 256 --gen-hidden 50 --latent-dim 128 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project mimic-irregular-A --random-seed $seed --drop 0.25
    # python run_exp.py --model-type 'mTAND' --alpha 5 --niters 300 --lr 0.0001 --batch-size 128 --rec-hidden 256 --gen-hidden 50 --latent-dim 128 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project mimic-irregular-A --random-seed $seed --drop 0.50
    # python run_exp.py --model-type 'mTAND' --alpha 5 --niters 300 --lr 0.0001 --batch-size 128 --rec-hidden 256 --gen-hidden 50 --latent-dim 128 --norm --kl --learn-emb --k-iwae 1 --dataset $data --project mimic-irregular-A --random-seed $seed --drop 0.75

    ####################################################
    # Perceiver
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --num-latents 48 --drop 0.25
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --num-latents 48 --drop 0.75

    ###################################################
    # TRANSFORMER
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --random-seed $seed --num-latents 48 --project mimic-irregular
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --random-seed $seed --num-latents 48 --project mimic-irregular --drop 0.25
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --random-seed $seed --num-latents 48 --project mimic-irregular --drop 0.50
    python run_exp.py --model-type 'TRANSFORMER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --random-seed $seed --num-latents 48 --project mimic-irregular --drop 0.75

    ######################################
    # COPER
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --num-latents 48
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --num-latents 48 --drop 0.25 
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --num-latents 48 --drop 0.50
    python run_exp.py --model-type 'COPER' --cont-in --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset $data --project mimic-irregular-A --random-seed $seed --num-latents 48 --drop 0.75
    
    ###################################
    # # LODE
    # python run_exp.py --model-type 'LODE'  --niters 300 --batch-size 32 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project mimic-irregular-A --random-seed $seed
    # python run_exp.py --model-type 'LODE'  --niters 300 --batch-size 32 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project mimic-irregular-A --random-seed $seed --drop 0.25
    # python run_exp.py --model-type 'LODE'  --niters 300 --batch-size 32 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project mimic-irregular-A --random-seed $seed --drop 0.50
    # python run_exp.py --model-type 'LODE'  --niters 300 --batch-size 32 --latent-dim 20 --rec-dims 40 --poisson --dataset $data --project mimic-irregular-A --random-seed $seed --drop 0.75
    
    #######################################
done


###############################################################################################################

conda deactivate
