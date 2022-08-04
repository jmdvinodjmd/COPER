#!/bin/bash
#SBATCH --time=1
eval "$(conda shell.bash hook)"
conda activate base
####################################
# LSTM
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 1
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 1 --drop 0.25
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 1 --drop 0.50
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 1 --drop 0.75

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 2
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 2 --drop 0.25
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 2 --drop 0.50
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 2 --drop 0.75

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 3
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 3 --drop 0.25
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 3 --drop 0.50
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --random-seed 3 --drop 0.75

# Perceiver
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 1
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 1 --drop 0.25 
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 1 --drop 0.50
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 1 --drop 0.75

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 2
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 2 --drop 0.25 
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 2 --drop 0.50
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 2 --drop 0.75

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 3
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 3 --drop 0.25 
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 3 --drop 0.50
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 3 --drop 0.75

# COPER
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 1
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 1 --drop 0.25 
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 1 --drop 0.50
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 1 --drop 0.75

python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 2
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 2 --drop 0.25 
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 2 --drop 0.50
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 2 --drop 0.75

python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 3
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 3 --drop 0.25 
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 3 --drop 0.50
python run_exp.py --model-type 'COPER' --cont-in --cont-out --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --random-seed 3 --drop 0.75

####################################

conda deactivate
