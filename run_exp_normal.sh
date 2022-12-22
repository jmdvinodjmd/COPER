##################################################################################
#################################################################################
### Run experiments for comparing Perceiver baselines
##################################################################################
# mimic dataset
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 1
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 2
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 3
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 4
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 5
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 6
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 7
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 8
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 9
python run_exp.py --model-type 'TCN' --dataset 'mimic' --project mimic-tcn --random-seed 10

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 1
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 2
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 3
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 4
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 5
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 6
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 7
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 8
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 9
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5  --dataset 'mimic' --project mimic-tcn --random-seed 10

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 1 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 2 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 3 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 4 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 5 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 6 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 7 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 8 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 9 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'mimic' --project mimic-tcn --random-seed 10 --num-latents 48

##################################################################################
## physionet dataset
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'LSTM' --latent-dim 50 --num-layers 2 --lstm-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 1
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 2
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 3
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 4
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 5

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 6
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 7
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 8
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 9
python run_exp.py --model-type 'TCN' --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 10

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 1 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 2 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 3 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 4 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 5 --num-latents 48

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 1 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 2 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 3 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 4 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 5 --num-latents 48

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 1 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 2 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 3 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 4 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 5 --num-latents 48

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 1 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 2 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 3 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 4 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 5 --num-latents 48

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 1 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 2 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 3 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 4 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 5 --num-latents 48

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 6 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 7 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 8 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 9 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 1 --project physionet-tcn --random-seed 10 --num-latents 48

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 6 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 7 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 8 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 9 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 2 --project physionet-tcn --random-seed 10 --num-latents 48

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 6 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 7 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 8 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 9 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 3 --project physionet-tcn --random-seed 10 --num-latents 48

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 6 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 7 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 8 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 9 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 4 --project physionet-tcn --random-seed 10 --num-latents 48

python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 6 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 7 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 8 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 9 --num-latents 48
python run_exp.py --model-type 'PERCEIVER' --self-per-cross-attn 1 --latent-heads 2 --cross-heads 1 --cross-dim-head 128 --latent-dim-head 128 --latent-dim 64 --units 128 --ode-dropout 0.5 --att-dropout 0.5 --ff-dropout 0.5 --dataset 'physionet' --fold 5 --project physionet-tcn --random-seed 10 --num-latents 48

##################################################################################
####################################